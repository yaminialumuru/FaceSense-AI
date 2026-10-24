// Pure logic shared by the browser app and the Node tests. No DOM access here.
export const DEFAULTS = {
  emotionConf: 0.5, earThresh: 0.2, marThresh: 0.5, smooth: 0.5, sirenEnabled: true, sirenSeconds: 5, stopOnClear: true, yawnMs: 1500, possibleYawnMs: 600,
  closureMs: 1000, blinkMaxMs: 400, perclosWindowMs: 30000, perclosLimit: 0.35,
  volume: 0.6, autoAlert: true, cooldownS: 15, recording: true, retentionDays: 30,
};
export const LIMITS = {
  emotionConf: [0.1, 0.99], earThresh: [0.05, 0.4], marThresh: [0.2, 1.5], yawnMs: [500, 10000],
  closureMs: [300, 10000], volume: [0, 1], smooth: [0.1, 1], sirenSeconds: [1, 30], cooldownS: [1, 300], retentionDays: [1, 365],
};
export function validateSettings(input) {
  const out = { ...DEFAULTS }, errors = [];
  for (const k of Object.keys(DEFAULTS)) {
    if (!(k in input)) continue;
    const v = input[k];
    if (typeof DEFAULTS[k] === 'boolean') { out[k] = !!v; continue; }
    const n = Number(v), lim = LIMITS[k];
    if (!Number.isFinite(n) || (lim && (n < lim[0] || n > lim[1]))) { errors.push(`${k} must be between ${lim ? lim[0] : '?'} and ${lim ? lim[1] : '?'}`); continue; }
    out[k] = n;
  }
  return { settings: out, errors };
}
// MediaPipe 478-point mesh indices.
const R_EYE = [33, 160, 158, 133, 153, 144], L_EYE = [362, 385, 387, 263, 373, 380];
const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function eyeAR(p, i) { return (d(p[i[1]], p[i[5]]) + d(p[i[2]], p[i[4]])) / (2 * d(p[i[0]], p[i[3]])); }
export function ear(p) { return (eyeAR(p, R_EYE) + eyeAR(p, L_EYE)) / 2; }
export function mar(p) { return d(p[13], p[14]) / d(p[78], p[308]); }
export function softmax(a) { const m = Math.max(...a), e = a.map(x => Math.exp(x - m)), s = e.reduce((x, y) => x + y, 0); return e.map(x => x / s); }
// Output order of emotion-ferplus-8.onnx (the model also has "contempt"; the app lists it separately).
export const FERPLUS = ['neutral', 'happy', 'surprised', 'sad', 'angry', 'disgusted', 'fearful', 'contempt'];
export function emotionResult(logits, minConf) {
  const probs = softmax(Array.from(logits)); let i = 0;
  probs.forEach((v, j) => { if (v > probs[i]) i = j; });
  return { label: FERPLUS[i], conf: probs[i], uncertain: probs[i] < minConf, probs };
}
export function stateColor(state) {
  if (state === 'alert' || state === 'possible_yawn' || state === 'possible_drowsy') return 'green';
  if (state === 'yawning' || state === 'eye_closure' || state === 'drowsy') return 'red';
  return 'gray'; // no_face, uncertain, unavailable
}
export class Detector {
  constructor(s) { this.s = s; this.reset(); }
  reset() { this.eyeStart = null; this.mouthStart = null; this.eyeFired = false; this.yawnFired = false; this.drowsy = false; this.samples = []; this.state = 'no_face'; this.sEar = null; this.sMar = null; }
  _close(t, ev) {
    if (this.eyeFired) ev.push({ kind: 'end', type: 'eye_closure', t, durationMs: t - this.eyeStart });
    if (this.yawnFired) ev.push({ kind: 'end', type: 'yawn', t, durationMs: t - this.mouthStart });
    if (this.drowsy) ev.push({ kind: 'end', type: 'drowsiness', t });
    this.eyeStart = this.mouthStart = null; this.eyeFired = this.yawnFired = this.drowsy = false;
  }
  update(t, m, uncertain = false) {
    const ev = [], s = this.s;
    if (!m || uncertain) { this._close(t, ev); this.samples = []; this.sEar = this.sMar = null; this.state = m ? 'uncertain' : 'no_face'; return { state: this.state, events: ev }; }
    const a = s.smooth ?? 1; // exponential smoothing against single-frame landmark jitter
    this.sEar = this.sEar == null ? m.ear : a * m.ear + (1 - a) * this.sEar; this.sMar = this.sMar == null ? m.mar : a * m.mar + (1 - a) * this.sMar;
    m = { ear: this.sEar, mar: this.sMar };
    const closed = m.ear < s.earThresh, open = m.mar > s.marThresh;
    if (closed) {
      if (this.eyeStart == null) this.eyeStart = t;
      if (!this.eyeFired && t - this.eyeStart >= s.closureMs) { this.eyeFired = true; ev.push({ kind: 'onset', type: 'eye_closure', t, evidence: { ear: +m.ear.toFixed(3), threshold: s.earThresh } }); }
    } else if (this.eyeStart != null) {
      if (this.eyeFired) ev.push({ kind: 'end', type: 'eye_closure', t, durationMs: t - this.eyeStart });
      this.eyeStart = null; this.eyeFired = false; // shorter closures (blinks) produce no event
    }
    if (open) {
      if (this.mouthStart == null) this.mouthStart = t;
      if (!this.yawnFired && t - this.mouthStart >= s.yawnMs) { this.yawnFired = true; ev.push({ kind: 'onset', type: 'yawn', t, evidence: { mar: +m.mar.toFixed(3), threshold: s.marThresh } }); }
    } else if (this.mouthStart != null) {
      if (this.yawnFired) ev.push({ kind: 'end', type: 'yawn', t, durationMs: t - this.mouthStart });
      this.mouthStart = null; this.yawnFired = false;
    }
    // PERCLOS-style drowsiness: share of closed frames over a window, only once half the window is observed.
    this.samples.push({ t, closed }); const w = s.perclosWindowMs;
    while (this.samples.length && t - this.samples[0].t > w) this.samples.shift();
    const covered = t - this.samples[0].t >= w / 2;
    const frac = this.samples.filter(x => x.closed).length / this.samples.length;
    const nowDrowsy = covered && frac >= (this.drowsy ? s.perclosLimit * 0.7 : s.perclosLimit); // hysteresis avoids flicker
    if (nowDrowsy && !this.drowsy) ev.push({ kind: 'onset', type: 'drowsiness', t, evidence: { perclos: +frac.toFixed(2), limit: s.perclosLimit } });
    if (!nowDrowsy && this.drowsy) ev.push({ kind: 'end', type: 'drowsiness', t });
    this.drowsy = nowDrowsy;
    this.state = this.drowsy ? 'drowsy' : this.eyeFired ? 'eye_closure' : this.yawnFired ? 'yawning'
      : (this.eyeStart != null && t - this.eyeStart >= s.blinkMaxMs) ? 'possible_drowsy'
      : (this.mouthStart != null && t - this.mouthStart >= s.possibleYawnMs) ? 'possible_yawn' : 'alert';
    return { state: this.state, events: ev, perclos: frac, ear: m.ear, mar: m.mar };
  }
}
export class AlarmGate { // one decision per detection event, plus a cooldown between alarms
  constructor() { this.last = -Infinity; }
  remaining(now, s) { return Math.max(0, this.last + s.cooldownS * 1000 - now); }
  allow(now, s) { if (!s.autoAlert || now - this.last < s.cooldownS * 1000) return false; this.last = now; return true; }
}
export class Tracker { // session-level temporary IDs by nearest centroid; not identity
  constructor(mk, maxDist = 0.15) { this.tracks = []; this.next = 1; this.mk = mk; this.maxDist = maxDist; }
  assign(cents, t) {
    const used = new Set(), out = [], lostEvents = [];
    for (const c of cents) {
      let best = null, bd = Infinity;
      for (const tr of this.tracks) if (!used.has(tr.id)) { const dd = Math.hypot(tr.x - c.x, tr.y - c.y); if (dd < bd) { bd = dd; best = tr; } }
      let tr = best && bd <= this.maxDist ? best : null;
      if (!tr) { tr = { id: this.next++, det: this.mk() }; this.tracks.push(tr); }
      tr.x = c.x; tr.y = c.y; tr.seen = t; used.add(tr.id); out.push(tr);
    }
    for (const tr of this.tracks) if (!used.has(tr.id)) for (const e of tr.det.update(t, null).events) lostEvents.push({ ...e, track: tr.id });
    this.tracks = this.tracks.filter(tr => t - tr.seen < 2000);
    return { tracks: out, lostEvents };
  }
}
const q = v => { v = v == null ? '' : String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
export function toCSV(header, rows) { return [header.join(','), ...rows.map(r => header.map(h => q(r[h])).join(','))].join('\r\n'); }
