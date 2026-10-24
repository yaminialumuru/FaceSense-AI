import * as E from './engine.js';
const $ = id => document.getElementById(id);
const VISION = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
const LS = { settings: 'fs.settings', sessions: 'fs.sessions' };
const load = (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } };
const save = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch { setStatus('Could not save: browser storage is full or blocked.'); return false; } };
const txt = (id, v) => { $(id).textContent = v; };
function setStatus(m) { txt('status', m); }

// ---------- settings ----------
let settings = E.validateSettings(load(LS.settings, {})).settings;
const FIELDS = [['emotionConf', 'Emotion confidence limit', 'number', .01], ['earThresh', 'Eye-closed below (EAR)', 'number', .01], ['marThresh', 'Mouth-open above (MAR)', 'number', .05],
  ['yawnMs', 'Sustained yawn after (ms)', 'number', 100], ['closureMs', 'Prolonged eye closure after (ms)', 'number', 100], ['volume', 'Siren volume (0 to 1)', 'number', .05],
  ['cooldownS', 'Alert cooldown (seconds)', 'number', 1], ['retentionDays', 'Keep sessions for (days)', 'number', 1], ['smooth', 'Smoothing (1 = none, lower = smoother)', 'number', .05], ['sirenSeconds', 'Siren length (seconds)', 'number', 1], ['sirenEnabled', 'Siren sound enabled', 'checkbox'], ['stopOnClear', 'Stop siren when the event ends', 'checkbox'], ['autoAlert', 'Automatic alerts in Driver Mode', 'checkbox'], ['recording', 'Record events and predictions', 'checkbox']];
function buildSettings() {
  const f = $('setForm'); f.textContent = '';
  for (const [k, label, type, step] of FIELDS) {
    const l = document.createElement('label'), i = document.createElement('input'); i.id = 's_' + k; i.type = type; if (step) i.step = step;
    if (type === 'checkbox') i.checked = settings[k]; else i.value = settings[k];
    l.append(i, ' ' + label); f.append(l);
  }
}
$('setSave').onclick = () => {
  const raw = {}; for (const [k, , t] of FIELDS) raw[k] = t === 'checkbox' ? $('s_' + k).checked : $('s_' + k).value;
  const r = E.validateSettings(raw);
  if (r.errors.length) { txt('setMsg', 'Not saved: ' + r.errors.join('; ')); return; }
  Object.assign(settings, r.settings); save(LS.settings, settings); txt('setMsg', 'Settings saved. They apply to the running detector right away.'); applyVolume();
};
$('setReset').onclick = () => { Object.assign(settings, E.DEFAULTS); save(LS.settings, settings); buildSettings(); txt('setMsg', 'Defaults restored.'); applyVolume(); };

// ---------- siren ----------
let actx = null, siren = null, sirenTimer = null; const gate = new E.AlarmGate();
function applyVolume() { if (siren) siren.g.gain.value = settings.volume; $('vol').value = settings.volume; }
async function enableAudio() {
  try { actx = actx || new AudioContext(); await actx.resume(); } catch (e) { txt('audioState', 'Audio blocked by the browser: ' + e.message); return false; }
  txt('audioState', actx.state === 'running' ? 'Audio is on.' : 'Audio is blocked. Click the page and try again.'); return actx.state === 'running';
}
function startSiren() {
  if (!settings.sirenEnabled) { txt('audioState', 'Siren sound is turned off in Settings.'); return false; }
  if (!actx || actx.state !== 'running') { txt('audioState', 'Audio is off. Press "Turn audio on" first.'); return false; }
  stopSiren(); const o = actx.createOscillator(), l = actx.createOscillator(), lg = actx.createGain(), g = actx.createGain();
  o.type = 'sawtooth'; o.frequency.value = 800; l.frequency.value = 2; lg.gain.value = 300; g.gain.value = settings.volume;
  l.connect(lg); lg.connect(o.frequency); o.connect(g); g.connect(actx.destination); o.start(); l.start(); siren = { o, l, g };
  sirenTimer = setTimeout(stopSiren, settings.sirenSeconds * 1000); return true;
}
function stopSiren() { clearTimeout(sirenTimer); if (siren) { try { siren.o.stop(); siren.l.stop(); } catch {} siren = null; } }
$('audioOn').onclick = enableAudio; $('sirenTest').onclick = async () => { if (!actx || actx.state !== 'running') await enableAudio(); const sv = settings.sirenEnabled; settings.sirenEnabled = true; if (!startSiren()) txt('audioState', 'Test failed: the browser blocked audio. Click the page and try again.'); settings.sirenEnabled = sv; }; $('sirenStop').onclick = stopSiren;
$('vol').oninput = e => { settings.volume = +e.target.value; save(LS.settings, settings); applyVolume(); };

// ---------- models ----------
const M = { landmarker: null, ort: null, emo: null, lmState: 'not loaded', emoState: 'not loaded' };
function showModels() { txt('modelState', `Face landmarks: ${M.lmState}. Expression model: ${M.emoState}.`); }
const loadScript = src => new Promise((ok, no) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = () => no(new Error('could not load ' + src)); document.head.append(s); });
async function loadModels() {
  if (!M.landmarker) {
    M.lmState = 'loading…'; showModels();
    try {
      const v = await import(VISION + '/vision_bundle.mjs'); const fs = await v.FilesetResolver.forVisionTasks(VISION + '/wasm');
      const mk = d => v.FaceLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: 'models/face_landmarker.task', delegate: d }, runningMode: 'VIDEO', numFaces: 4 });
      try { M.landmarker = await mk('GPU'); } catch { M.landmarker = await mk('CPU'); }
      M.lmState = 'ready';
    } catch (e) { M.lmState = 'UNAVAILABLE (' + e.message + '). Run scripts\\download-models.ps1 and check your internet connection.'; }
    showModels();
  }
  if (!M.emo && M.emoState !== 'UNAVAILABLE') {
    M.emoState = 'loading…'; showModels();
    try {
      await loadScript(ORT + 'ort.min.js'); window.ort.env.wasm.wasmPaths = ORT;
      M.emo = await window.ort.InferenceSession.create('models/emotion-ferplus-8.onnx'); M.emoState = 'ready (FER+, 64x64 grayscale)';
    } catch (e) { M.emoState = 'UNAVAILABLE (' + e.message + '). Expression predictions are off; eye and yawn detection still work.'; }
    showModels();
  }
  return !!M.landmarker;
}
const crop = document.createElement('canvas'); crop.width = crop.height = 64; const cctx = crop.getContext('2d', { willReadFrequently: true });
async function predictEmotion(video, box) {
  const W = video.videoWidth, H = video.videoHeight, pad = .15, bw = (box.x1 - box.x0) * W, bh = (box.y1 - box.y0) * H;
  const sx = Math.max(0, box.x0 * W - bw * pad), sy = Math.max(0, box.y0 * H - bh * pad), sw = Math.min(W - sx, bw * (1 + 2 * pad)), sh = Math.min(H - sy, bh * (1 + 2 * pad));
  cctx.drawImage(video, sx, sy, sw, sh, 0, 0, 64, 64); const px = cctx.getImageData(0, 0, 64, 64).data, f = new Float32Array(4096);
  for (let i = 0; i < 4096; i++) f[i] = .299 * px[i * 4] + .587 * px[i * 4 + 1] + .114 * px[i * 4 + 2]; // raw 0-255 grayscale, as the model expects
  const out = await M.emo.run({ [M.emo.inputNames[0]]: new window.ort.Tensor('float32', f, [1, 1, 64, 64]) });
  return E.emotionResult(out[M.emo.outputNames[0]].data, settings.emotionConf);
}

// ---------- session ----------
let sess = null, running = false, paused = false, stream = null, raf = 0, tracker = null, lastEmo = 0, emoBusy = false, lastVT = -1, curEmo = null, openEv = {};
const sessions = () => load(LS.sessions, []);
function purge() { const cut = Date.now() - settings.retentionDays * 864e5; const all = sessions(), keep = all.filter(s => new Date(s.start).getTime() >= cut); if (keep.length !== all.length) save(LS.sessions, keep); }
const mode = () => ['driver', 'study', 'expr'].includes(view) ? view : 'general';
function startSession() {
  sess = { id: 's' + Date.now().toString(36), mode: mode(), start: new Date().toISOString(), end: null, config: { ...settings }, models: { landmarks: M.lmState, expression: M.emoState }, events: [], alerts: [], emotionCounts: {}, predictions: [], maxFaces: 0 };
  tracker = new E.Tracker(() => new E.Detector(settings)); openEv = {};
}
function endSession() {
  if (!sess) return; sess.end = new Date().toISOString();
  if (settings.recording) { const all = sessions(); all.unshift(sess); save(LS.sessions, all.slice(0, 200)); } else setStatus('Session ended. Recording is off, so nothing was saved.');
  const sm = ` Last session: ${Math.round((Date.parse(sess.end) - Date.parse(sess.start)) / 1000)} s, up to ${sess.maxFaces} face(s), ${count('yawn')} yawn event(s), ${count('eye_closure') + count('drowsiness')} drowsiness event(s), ${sess.alerts.length} alert(s).`; txt('sum-last', sm);
  sess = null; renderDash(); if (view === 'behavior') renderBehavior();
}
const count = t => sess ? sess.events.filter(e => e.type === t).length : 0;
let lastAlert = null, blocked = '';
const REASON = { yawn: 'Yawning detected (sustained)', eye_closure: 'Drowsiness detected (prolonged eye closure)', drowsiness: 'Drowsiness detected (eyes closed too often)' };
function handleEvents(evs, trackId, now) {
  for (const e of evs) {
    const tid = e.track ?? trackId, key = tid + e.type;
    if (e.kind === 'onset') {
      const rec = { track: tid, type: e.type, start: new Date().toISOString(), end: '', durationMs: '', evidence: JSON.stringify(e.evidence || {}) };
      sess.events.push(rec); openEv[key] = rec; console.debug('[facesense] event', e.type, 'face', tid, e.evidence);
      if (view !== 'driver') blocked = 'No siren: Driver Mode is not open.';
      else if (!settings.autoAlert) blocked = 'No siren: automatic alerts are off in Settings.';
      else if (!gate.allow(now, settings)) blocked = `No siren: cooldown (${Math.ceil(gate.remaining(now, settings) / 1000)} s left).`;
      else {
        const al = { t: new Date().toISOString(), reason: REASON[e.type], face: tid, siren: '' }; sess.alerts.push(al); lastAlert = al; blocked = '';
        al.siren = !settings.sirenEnabled ? 'sound disabled in Settings' : startSiren() ? 'playing' : 'audio not enabled or blocked';
        $('warn').hidden = false; txt('warn', `${al.reason}. Siren: ${al.siren}.`); clearTimeout(handleEvents.t); handleEvents.t = setTimeout(() => { $('warn').hidden = true; }, 6000);
      }
    } else if (openEv[key]) {
      openEv[key].end = new Date().toISOString(); openEv[key].durationMs = Math.round(e.durationMs ?? (Date.parse(openEv[key].end) - Date.parse(openEv[key].start))); delete openEv[key];
      if (settings.stopOnClear && !Object.keys(openEv).length) stopSiren();
    }
  }
}
const COL = { green: '#1BA784', red: '#D6336C', gray: '#8D8AA0' };
const LABEL = { alert: 'Alert / normal pattern', possible_yawn: 'Possible yawn', yawning: 'Yawning detected', possible_drowsy: 'Possible drowsiness', eye_closure: 'Drowsiness detected (eyes closed)', drowsy: 'Drowsiness detected', no_face: 'No face detected', uncertain: 'Uncertain / unavailable' };
const EMOJI = { neutral: '😐', happy: '😊', surprised: '😲', sad: '😢', angry: '😠', disgusted: '🤢', fearful: '😨', contempt: '😒' };
const cap = w => w[0].toUpperCase() + w.slice(1);
function emoText(tr) { // expression label for one face; behavioural state is shown separately
  if (!M.emo) return 'Expression model unavailable'; if (!tr || !tr.emo) return 'Analyzing…';
  const e = tr.emo, pc = Math.round(e.conf * 100); return e.uncertain ? `Uncertain (${pc}%)` : `${EMOJI[e.label]} ${cap(e.label)} ${pc}%`;
}
function frame() {
  raf = requestAnimationFrame(frame); const v = $('vid');
  if (paused || !M.landmarker || !sess || v.readyState < 2 || v.currentTime === lastVT) return; lastVT = v.currentTime;
  const now = performance.now(); let res; try { res = M.landmarker.detectForVideo(v, now); } catch (e) { setStatus('Inference error: ' + e.message); return; }
  const cv = $('ov'), c = cv.getContext('2d'), W = cv.width, H = cv.height; c.clearRect(0, 0, W, H);
  const faces = (res.faceLandmarks || []).map(p => { let x0 = 1, y0 = 1, x1 = 0, y1 = 0; for (const q of p) { x0 = Math.min(x0, q.x); y0 = Math.min(y0, q.y); x1 = Math.max(x1, q.x); y1 = Math.max(y1, q.y); } return { p, box: { x0, y0, x1, y1 }, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 }; });
  const { tracks, lostEvents } = tracker.assign(faces.map(f => ({ x: f.cx, y: f.cy })), now); handleEvents(lostEvents, 'x', now);
  sess.maxFaces = Math.max(sess.maxFaces, faces.length); txt('st-faces', faces.length); let primary = null, pa = 0;
  faces.forEach((f, i) => { // 1) behaviour from landmarks
    const tr = tracks[i]; tr.f = f; const area = (f.box.x1 - f.box.x0) * (f.box.y1 - f.box.y0); if (area > pa) { pa = area; primary = tr; }
    const r = tr.det.update(now, { ear: E.ear(f.p), mar: E.mar(f.p) }, f.box.x1 - f.box.x0 < .1); tr.m = { ear: r.ear, mar: r.mar }; tr.state = r.state; tr.perclos = r.perclos; handleEvents(r.events, tr.id, now);
  });
  if (M.emo && !emoBusy && now - lastEmo > 250 && tracks.length) { // 2) expression: one face per tick, oldest result first; the result is bound to that face's track
    const tr = tracks.reduce((a, b) => ((a.emoT || 0) <= (b.emoT || 0) ? a : b)); lastEmo = now; emoBusy = true; tr.analyzing = true; const box = { ...tr.f.box };
    predictEmotion(v, box).then(r => { tr.emo = r; tr.emoT = performance.now(); recordEmotion(r, tr.id); })
      .catch(e => { M.emoState = 'UNAVAILABLE (' + e.message + ')'; M.emo = null; showModels(); }).finally(() => { emoBusy = false; tr.analyzing = false; });
  }
  c.font = '600 ' + Math.max(14, W / 48) + 'px Figtree,sans-serif'; const fh = Math.max(14, W / 48) + 10;
  tracks.forEach(tr => { // 3) draw: box colour = behaviour; top label = expression; bottom tag = behaviour
    const f = tr.f, col = COL[E.stateColor(tr.state)], x = (1 - f.box.x1) * W, y = f.box.y0 * H, w = (f.box.x1 - f.box.x0) * W, h = (f.box.y1 - f.box.y0) * H;
    c.lineWidth = 4; c.strokeStyle = col; c.strokeRect(x, y, w, h);
    const top = `#${tr.id}  ${emoText(tr)}`, tw = c.measureText(top).width + 12, ty = Math.max(0, y - fh); c.fillStyle = '#2A1740'; c.fillRect(x, ty, tw, fh); c.fillStyle = '#fff'; c.fillText(top, x + 6, ty + fh - 8);
    const bot = LABEL[tr.state], bw = c.measureText(bot).width + 12, by = Math.min(H - fh, y + h); c.fillStyle = col; c.fillRect(x, by, bw, fh); c.fillStyle = '#fff'; c.fillText(bot, x + 6, by + fh - 8);
  });
  updateLive(primary, tracks.length, now);
}
function recordEmotion(r, track) {
  if (!sess) return; const lab = r.uncertain ? 'uncertain' : r.label; sess.emotionCounts[lab] = (sess.emotionCounts[lab] || 0) + 1;
  if (sess.predictions.length < 1500) sess.predictions.push({ t: new Date().toISOString(), track, label: r.label, confidence: +r.conf.toFixed(3), uncertain: r.uncertain });
}
// ---------- live UI ----------
let lastDiag = 0;
function updateLive(tr, nFaces, now) {
  const st = tr ? tr.state : 'no_face'; txt('st-yawn', count('yawn')); txt('st-drowsy', count('drowsiness') + count('eye_closure')); txt('st-alert', sess ? sess.alerts.length : 0);
  for (const id of ['d-state', 's-state', 'dash-state']) txt(id, LABEL[st]);
  txt('d-ear', tr ? tr.m.ear.toFixed(2) + ` (closed below ${settings.earThresh})` : '–'); txt('d-mar', tr ? tr.m.mar.toFixed(2) + ` (open above ${settings.marThresh})` : '–');
  txt('d-perclos', tr && tr.perclos != null ? Math.round(tr.perclos * 100) + '% of the last 30 s' : '–');
  for (const id of ['d-emo', 'x-emo', 'dash-emo']) txt(id, tr ? emoText(tr) : '–');
  const e = tr && tr.emo; $('x-bars').textContent = ''; if (e) E.FERPLUS.forEach((n, i) => { const r = document.createElement('div'); r.className = 'bar'; const b = document.createElement('i'); b.style.width = (e.probs[i] * 100).toFixed(1) + '%'; r.append(cap(n) + ' ' + Math.round(e.probs[i] * 100) + '%', b); $('x-bars').append(r); });
  txt('d-audio', !actx ? 'Off. Press Enable audio' : actx.state === 'running' ? (settings.sirenEnabled ? 'On' : 'On, siren disabled in Settings') : 'Blocked by browser');
  txt('d-alert', !settings.autoAlert ? 'Off in Settings' : lastAlert ? `${lastAlert.reason}, siren ${lastAlert.siren}` : 'Armed, waiting for an event');
  txt('d-cool', gate.remaining(now, settings) > 0 ? Math.ceil(gate.remaining(now, settings) / 1000) + ' s left' : 'Ready');
  if (sess) { const t = Math.floor((Date.now() - Date.parse(sess.start)) / 1000); txt('timer', `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`); }
  const list = sess ? sess.events.slice(-6).reverse() : []; for (const id of ['d-events', 's-events']) { $(id).textContent = ''; if (!list.length) $(id).append(li('No events yet.')); list.forEach(x => $(id).append(li(`${new Date(x.start).toLocaleTimeString()}  face #${x.track}  ${x.type.replace('_', ' ')}${x.durationMs ? ' (' + (x.durationMs / 1000).toFixed(1) + ' s)' : ' (ongoing)'}`))); }
  if (view === 'expr') { $('x-hist').textContent = ''; (sess?.predictions || []).slice(-8).reverse().forEach(p => $('x-hist').append(li(`${new Date(p.t).toLocaleTimeString()} face #${p.track}: ${p.uncertain ? 'uncertain (best guess ' + p.label + ')' : p.label} ${Math.round(p.confidence * 100)}%`))); }
  if (view === 'driver' && now - lastDiag > 200) { lastDiag = now; const d = tr && tr.det, m = tr && tr.m;
    txt('diag-pre', !tr ? `faces: ${nFaces}\nno face in view, detection idle\naudio: ${actx ? actx.state : 'not created'}  cooldown: ${$('d-cool').textContent}\nlast block reason: ${blocked || '-'}` : [
      `pipeline: landmarks=${M.lmState.slice(0, 20)} faces=${nFaces} expression=${M.emo ? 'ready' : 'off'}`,
      `MAR ${m.mar.toFixed(3)}  threshold ${settings.marThresh}  mouth ${d.mouthStart != null ? 'OPEN ' + Math.round(now - d.mouthStart) + ' ms (yawn at ' + settings.yawnMs + ' ms)' : 'closed'}`,
      `EAR ${m.ear.toFixed(3)}  threshold ${settings.earThresh}  eyes ${d.eyeStart != null ? 'CLOSED ' + Math.round(now - d.eyeStart) + ' ms (long closure at ' + settings.closureMs + ' ms)' : 'open'}`,
      `eyes-closed share ${Math.round((tr.perclos || 0) * 100)}% (drowsy at ${Math.round(settings.perclosLimit * 100)}%)`,
      `state: ${tr.state}   yawnEvent=${d.yawnFired}  closureEvent=${d.eyeFired}  drowsyEvent=${d.drowsy}`,
      `audio: context=${actx ? actx.state : 'not created'} sirenEnabled=${settings.sirenEnabled} playing=${!!siren}`,
      `alerts: auto=${settings.autoAlert} cooldown=${$('d-cool').textContent} view=${view} session=${sess.mode}`,
      `last alert: ${lastAlert ? lastAlert.reason + ' / siren ' + lastAlert.siren : '-'}   last block reason: ${blocked || '-'}`].join('\n')); }
}
const li = t => { const e = document.createElement('li'); e.textContent = t; return e; };

// ---------- camera ----------
async function start() {
  if (running) return; enableAudio(); setStatus('Requesting camera…'); // audio must be unlocked inside this click
  if (!navigator.mediaDevices?.getUserMedia) { setStatus('This browser cannot open a camera. Use a current Chrome, Edge, Firefox or Safari on localhost or HTTPS.'); return; }
  try { stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } }); }
  catch (e) { setStatus(e.name === 'NotAllowedError' ? 'Camera access was blocked. Allow it in the site settings and try again.' : 'No camera found, or another app is using it.'); return; }
  const v = $('vid'); v.srcObject = stream; await v.play().catch(() => {}); $('camMsg').hidden = true; v.hidden = false;
  v.onloadedmetadata = null; $('ov').width = v.videoWidth; $('ov').height = v.videoHeight; $('cam').style.aspectRatio = v.videoWidth + '/' + v.videoHeight;
  setStatus('Loading models…'); const ok = await loadModels();
  if (!ok) { setStatus('Face landmark model failed to load, so detection is unavailable. See the model status below.'); stopCamera(); return; }
  startSession(); running = true; paused = false; $('btnStart').hidden = true; $('btnStop').hidden = false; $('btnPause').hidden = false; $('btnPause').textContent = 'Pause';
  setStatus(`Monitoring in ${sess.mode} mode. Video is processed on this device only.`); raf = requestAnimationFrame(frame);
}
function stopCamera() { cancelAnimationFrame(raf); stream?.getTracks().forEach(t => t.stop()); stream = null; const v = $('vid'); v.srcObject = null; v.hidden = true; $('camMsg').hidden = false; $('ov').getContext('2d').clearRect(0, 0, 9999, 9999); }
function stop() { stopSiren(); if (running) { running = false; endSession(); setStatus('Stopped. Camera released.'); } stopCamera(); $('btnStart').hidden = false; $('btnStop').hidden = true; $('btnPause').hidden = true; txt('st-faces', 0); }
$('btnStart').onclick = start; $('btnStop').onclick = stop;
$('btnPause').onclick = () => { paused = !paused; if (!paused) tracker.tracks.forEach(t => t.det.reset()); $('btnPause').textContent = paused ? 'Resume' : 'Pause'; setStatus(paused ? 'Paused. The camera is still on.' : 'Monitoring resumed.'); if (paused) $('ov').getContext('2d').clearRect(0, 0, 9999, 9999); };
addEventListener('pagehide', () => { if (running) { endSession(); } });

// ---------- views ----------
let view = 'dashboard';
const VIEWS = { dashboard: 'Dashboard', driver: 'Driver Mode', study: 'Study Mode', expr: 'Expressions', dist: 'Emotion Distribution', behavior: 'Behavior Analytics', settings: 'Settings' };
const side = [...document.querySelectorAll('.side button')];
side.forEach(b => { b.onclick = () => show(Object.keys(VIEWS).find(k => VIEWS[k] === b.textContent)); });
function show(v) { view = v; side.forEach(b => { if (VIEWS[v] === b.textContent) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current'); });
  for (const k in VIEWS) $('v-' + k).hidden = k !== v; $('camBlock').hidden = ['dist', 'behavior', 'settings'].includes(v);
  if (v === 'behavior') renderBehavior(); if (v === 'dashboard') renderDash(); if (v === 'dist') renderDist(); }
function renderDash() { const l = $('dash-list'); l.textContent = ''; const a = sessions().slice(0, 5); if (!a.length) l.append(li('No sessions yet. Start Driver Mode or Study Mode and your first session will appear here.')); a.forEach(x => l.append(li(`${new Date(x.start).toLocaleString()}  ${x.mode}  ${x.events.length} events, ${x.alerts.length} alerts`))); }
const SVGNS = 'http://www.w3.org/2000/svg';
function svgEl(tag, attrs, parent, text) { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; parent?.append(e); return e; }
// ----- Emotion Distribution -----
const ECOL = { neutral: '#8D8AA0', happy: '#FFC24B', surprised: '#6FD3BF', sad: '#5B7DB1', angry: '#D6336C', disgusted: '#8A9A2B', fearful: '#7A3FF2', contempt: '#B5651D' };
function allSessionsForDist() { const a = sessions(); if (sess) a.unshift({ ...sess, id: sess.id, live: true }); return a; }
let distBuilt = false;
function buildDistFilters() {
  const c = $('f-cats'); c.textContent = ''; E.FERPLUS.forEach(n => { const l = document.createElement('label'), i = document.createElement('input'); i.type = 'checkbox'; i.checked = true; i.value = n; i.onchange = renderDist; l.append(i, ' ' + EMOJI[n] + ' ' + cap(n)); c.append(l); });
  ['f-session', 'f-from', 'f-to', 'f-unc'].forEach(id => $(id).onchange = renderDist); distBuilt = true;
}
function renderDist() {
  if (!distBuilt) buildDistFilters();
  const ss = allSessionsForDist(), sel = $('f-session'), cur = sel.value; sel.textContent = '';
  const o0 = document.createElement('option'); o0.value = ''; o0.textContent = 'All sessions'; sel.append(o0);
  ss.forEach(x => { const o = document.createElement('option'); o.value = x.id; o.textContent = (x.live ? 'Live session, ' : '') + new Date(x.start).toLocaleString() + ' (' + x.mode + ')'; sel.append(o); }); sel.value = [...sel.options].some(o => o.value === cur) ? cur : '';
  const cats = new Set([...$('f-cats').querySelectorAll('input:checked')].map(i => i.value)), from = $('f-from').value ? Date.parse($('f-from').value) : -Infinity, to = $('f-to').value ? Date.parse($('f-to').value) + 864e5 : Infinity, unc = $('f-unc').checked;
  const counts = Object.fromEntries(E.FERPLUS.map(n => [n, 0])), per = []; let total = 0, skipped = 0;
  ss.filter(x => !sel.value || x.id === sel.value).forEach(x => { const c = {}; let n = 0;
    (x.predictions || []).forEach(p => { const t = Date.parse(p.t); if (t < from || t >= to || !cats.has(p.label)) return; if (p.uncertain && !unc) { skipped++; return; } counts[p.label]++; c[p.label] = (c[p.label] || 0) + 1; n++; total++; });
    if (n) per.push({ x, n, top: Object.entries(c).sort((a, b) => b[1] - a[1])[0] }); });
  $('dist-empty').hidden = total > 0; $('dist-body').hidden = total === 0; if (!total) return;
  const rows = E.FERPLUS.filter(n => counts[n] > 0).map(n => ({ n, v: counts[n], pct: counts[n] / total * 100 })).sort((a, b) => b.v - a.v);
  txt('dist-total', total); txt('dist-top', `${EMOJI[rows[0].n]} ${cap(rows[0].n)} (${rows[0].pct.toFixed(1)}%)`); txt('dist-note', skipped ? `${skipped} uncertain predictions are left out. Tick "Include uncertain" to count them by best guess.` : '');
  const pie = $('pie'); pie.textContent = ''; let ang = -Math.PI / 2; const R = 90;
  if (rows.length === 1) svgEl('circle', { cx: 100, cy: 100, r: R, fill: ECOL[rows[0].n] }, pie);
  else rows.forEach(r => { const a2 = ang + r.v / total * 2 * Math.PI, big = a2 - ang > Math.PI ? 1 : 0; svgEl('path', { d: `M100 100 L${100 + R * Math.cos(ang)} ${100 + R * Math.sin(ang)} A${R} ${R} 0 ${big} 1 ${100 + R * Math.cos(a2)} ${100 + R * Math.sin(a2)} Z`, fill: ECOL[r.n], stroke: 'var(--card)', 'stroke-width': 2 }, pie).append(Object.assign(svgEl('title', {}, null, `${cap(r.n)}: ${r.v} (${r.pct.toFixed(1)}%)`))); ang = a2; });
  const lg = $('legend'); lg.textContent = ''; rows.forEach(r => { const d = document.createElement('div'); const sw = document.createElement('span'); sw.className = 'sw'; sw.style.background = ECOL[r.n]; d.append(sw, `${EMOJI[r.n]} ${cap(r.n)}: ${r.v} (${r.pct.toFixed(1)}%)`); lg.append(d); });
  const bar = $('bars'); bar.textContent = ''; const max = Math.max(...rows.map(r => r.v)), bw = 360 / rows.length;
  rows.forEach((r, i) => { const h = r.v / max * 150, x = 30 + i * bw; svgEl('rect', { x: x + 6, y: 170 - h, width: Math.max(8, bw - 12), height: h, rx: 6, fill: ECOL[r.n] }, bar); svgEl('text', { x: x + bw / 2, y: 164 - h, 'text-anchor': 'middle', 'font-size': 11, fill: 'currentColor' }, bar, r.v); svgEl('text', { x: x + bw / 2, y: 188, 'text-anchor': 'middle', 'font-size': 10, fill: 'currentColor' }, bar, cap(r.n).slice(0, 9)); });
  svgEl('line', { x1: 30, y1: 170, x2: 395, y2: 170, stroke: 'currentColor', opacity: .3 }, bar);
  const tb = $('dist-per'); tb.textContent = ''; per.forEach(p => { const tr = document.createElement('tr'); [new Date(p.x.start).toLocaleString() + (p.x.live ? ' (live)' : ''), p.x.mode, p.n, `${cap(p.top[0])} (${p.top[1]})`].forEach(v => { const td = document.createElement('td'); td.textContent = v; tr.append(td); }); tb.append(tr); });
}
// ----- Behavior Analytics (session history, charts, export) -----
let sel = null;
function renderBehavior() {
  const all = sessions(), q = $('r-filter').value, rows = all.filter(s => !q || s.mode === q);
  const y = rows.reduce((a, s) => a + s.events.filter(e => e.type === 'yawn').length, 0), dr = rows.reduce((a, s) => a + s.events.filter(e => e.type !== 'yawn').length, 0), al = rows.reduce((a, s) => a + s.alerts.length, 0), sec = rows.reduce((a, s) => a + (Date.parse(s.end) - Date.parse(s.start)) / 1000, 0);
  txt('b-yawn', y); txt('b-drowsy', dr); txt('b-alerts', al); txt('b-time', Math.round(sec / 60) + ' min');
  $('b-empty').hidden = rows.length > 0; $('b-body').hidden = !rows.length; if (!rows.length) return;
  const ch = $('b-chart'); ch.textContent = ''; const last = rows.slice(0, 10).reverse(), mx = Math.max(1, ...last.map(s => Math.max(s.events.filter(e => e.type === 'yawn').length, s.events.filter(e => e.type !== 'yawn').length, s.alerts.length))), gw = 400 / last.length;
  last.forEach((s, i) => { const vals = [[s.events.filter(e => e.type === 'yawn').length, '#7A3FF2'], [s.events.filter(e => e.type !== 'yawn').length, '#D6336C'], [s.alerts.length, '#FFC24B']], x0 = 30 + i * gw;
    vals.forEach(([v, c], k) => { const h = v / mx * 130, w = Math.max(5, gw / 4); svgEl('rect', { x: x0 + 4 + k * (w + 2), y: 150 - h, width: w, height: h, rx: 3, fill: c }, ch).append(svgEl('title', {}, null, v)); });
    svgEl('text', { x: x0 + gw / 2, y: 168, 'text-anchor': 'middle', 'font-size': 9, fill: 'currentColor' }, ch, new Date(s.start).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' }) + ' ' + new Date(s.start).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })); });
  svgEl('line', { x1: 30, y1: 150, x2: 435, y2: 150, stroke: 'currentColor', opacity: .3 }, ch);
  const body = $('r-body'); body.textContent = '';
  rows.forEach(s => { const tr = document.createElement('tr'), dur = Math.round((Date.parse(s.end) - Date.parse(s.start)) / 1000);
    [new Date(s.start).toLocaleString(), s.mode, dur + ' s', s.events.filter(e => e.type === 'yawn').length, s.events.filter(e => e.type !== 'yawn').length, s.alerts.length].forEach(v => { const td = document.createElement('td'); td.textContent = v; tr.append(td); });
    const td = document.createElement('td'), a = document.createElement('button'), b = document.createElement('button'); a.textContent = 'Open'; b.textContent = 'Delete'; a.className = b.className = 'mini';
    a.onclick = () => { sel = s.id; renderDetail(); }; b.onclick = () => { if (confirm('Delete this session?')) { save(LS.sessions, sessions().filter(x => x.id !== s.id)); if (sel === s.id) sel = null; renderBehavior(); } }; td.append(a, b); tr.append(td); body.append(tr); });
  renderDetail();
}
function renderDetail() {
  const s = sessions().find(x => x.id === sel), d = $('r-detail'); d.hidden = !s; if (!s) return; d.textContent = '';
  const h = document.createElement('h3'); h.textContent = `Session ${s.id} (${s.mode})`; d.append(h);
  const p = document.createElement('p'); p.textContent = `Started ${new Date(s.start).toLocaleString()}. Most faces at once: ${s.maxFaces}. Models: landmarks ${s.models.landmarks}; expression ${s.models.expression}. Thresholds: EAR ${s.config.earThresh}, MAR ${s.config.marThresh}, yawn ${s.config.yawnMs} ms, closure ${s.config.closureMs} ms.`; d.append(p);
  const h3 = document.createElement('h3'); h3.textContent = 'Events (estimates, not diagnoses)'; d.append(h3); const ul = document.createElement('ul'); ul.className = 'plain'; if (!s.events.length) ul.append(li('No events.')); s.events.forEach(e => ul.append(li(`${new Date(e.start).toLocaleTimeString()} face #${e.track} ${e.type.replace('_', ' ')} ${e.durationMs ? (e.durationMs / 1000).toFixed(1) + ' s' : ''}`))); d.append(ul);
  const h4 = document.createElement('h3'); h4.textContent = 'Alerts'; d.append(h4); const ua = document.createElement('ul'); ua.className = 'plain'; if (!s.alerts.length) ua.append(li('No alerts.')); s.alerts.forEach(a => ua.append(li(`${new Date(a.t).toLocaleTimeString()} ${a.reason}, siren: ${a.siren || 'n/a'}`))); d.append(ua);
  const pr = document.createElement('button'); pr.className = 'mini'; pr.textContent = 'Print report'; pr.onclick = () => print(); d.append(pr);
}
$('r-filter').onchange = renderBehavior;
function dl(name, text) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv' })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); }
$('r-csv').onclick = () => { const rows = []; sessions().forEach(s => s.events.forEach(e => rows.push({ session: s.id, mode: s.mode, face_track: e.track, type: e.type, start: e.start, end: e.end, duration_ms: e.durationMs, evidence: e.evidence }))); if (!rows.length) { txt('r-msg', 'No events to export yet.'); return; } dl('facesense-events.csv', E.toCSV(['session', 'mode', 'face_track', 'type', 'start', 'end', 'duration_ms', 'evidence'], rows)); };
$('r-csv2').onclick = () => { const rows = []; sessions().forEach(s => s.predictions.forEach(p => rows.push({ session: s.id, time: p.t, face_track: p.track, label: p.label, confidence: p.confidence, uncertain: p.uncertain }))); if (!rows.length) { txt('r-msg', 'No predictions to export yet.'); return; } dl('facesense-predictions.csv', E.toCSV(['session', 'time', 'face_track', 'label', 'confidence', 'uncertain'], rows)); };
$('r-clear').onclick = () => { if (confirm('Delete all saved sessions?')) { save(LS.sessions, []); sel = null; renderBehavior(); } };
document.querySelectorAll('[data-go]').forEach(b => b.onclick = () => { show(b.dataset.go); document.getElementById('app').scrollIntoView(); });
purge(); buildSettings(); applyVolume(); showModels(); show('dashboard');
