import test from 'node:test'; import assert from 'node:assert/strict';
import * as E from '../engine.js';
const S = { ...E.DEFAULTS, smooth: 1 }; // raw values for the geometry tests; smoothing is tested separately
function face(eyeOpen, mouthOpen) {
  const p = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }));
  const eye = (idx, cx) => { p[idx[0]] = { x: cx - 1, y: 0 }; p[idx[3]] = { x: cx + 1, y: 0 };
    p[idx[1]] = { x: cx - .5, y: eyeOpen / 2 }; p[idx[5]] = { x: cx - .5, y: -eyeOpen / 2 };
    p[idx[2]] = { x: cx + .5, y: eyeOpen / 2 }; p[idx[4]] = { x: cx + .5, y: -eyeOpen / 2 }; };
  eye([33, 160, 158, 133, 153, 144], -3); eye([362, 385, 387, 263, 373, 380], 3);
  p[78] = { x: -1, y: 5 }; p[308] = { x: 1, y: 5 }; p[13] = { x: 0, y: 5 + mouthOpen / 2 }; p[14] = { x: 0, y: 5 - mouthOpen / 2 };
  return p;
}
const run = (det, ms, m, step = 50, t0 = 0) => { const ev = []; let r; for (let t = t0; t <= t0 + ms; t += step) { r = det.update(t, m); ev.push(...r.events); } return { ev, r }; };
test('EAR and MAR from landmarks', () => {
  assert.ok(Math.abs(E.ear(face(1, 0)) - 0.5) < 1e-9); assert.ok(E.ear(face(0.1, 0)) < 0.2);
  assert.ok(Math.abs(E.mar(face(1, 1.2)) - 0.6) < 1e-9);
});
test('blink is not an event; prolonged closure fires once and ends once', () => {
  const open = { ear: .3, mar: .1 }, shut = { ear: .1, mar: .1 };
  const d = new E.Detector(S); let a = run(d, 300, shut); assert.equal(a.ev.length, 0);
  run(d, 500, open, 50, 400);
  const d2 = new E.Detector(S); a = run(d2, 2000, shut);
  assert.equal(a.ev.filter(e => e.type === 'eye_closure' && e.kind === 'onset').length, 1);
  assert.equal(a.r.state, 'eye_closure');
  const b = run(d2, 100, open, 50, 2100);
  assert.equal(b.ev.filter(e => e.type === 'eye_closure' && e.kind === 'end').length, 1);
});
test('yawn: brief mouth opening ignored; sustained one fires once', () => {
  let d = new E.Detector(S); assert.equal(run(d, 800, { ear: .3, mar: .9 }).ev.length, 0);
  assert.equal(d.state, 'possible_yawn');
  d = new E.Detector(S); const a = run(d, 3000, { ear: .3, mar: .9 });
  assert.equal(a.ev.filter(e => e.type === 'yawn').length, 1); assert.equal(a.r.state, 'yawning');
});
test('drowsiness from repeated closures over the window (PERCLOS)', () => {
  const d = new E.Detector(S); let ev = [], t = 0;
  while (t < 40000) { const closed = (t % 2000) < 700; ev.push(...d.update(t, { ear: closed ? .1 : .3, mar: .1 }).events); t += 100; }
  assert.equal(ev.filter(e => e.type === 'drowsiness' && e.kind === 'onset').length, 1);
  assert.equal(ev.filter(e => e.type === 'eye_closure').length, 0);
});
test('no face and uncertain end open events and give gray state', () => {
  const d = new E.Detector(S); run(d, 2000, { ear: .1, mar: .1 });
  const r = d.update(2100, null); assert.equal(r.state, 'no_face'); assert.equal(r.events[0].kind, 'end');
  assert.equal(d.update(2200, { ear: .3, mar: .1 }, true).state, 'uncertain');
});
test('alarm gate: cooldown, disabled', () => {
  const g = new E.AlarmGate(); assert.ok(g.allow(0, S)); assert.ok(!g.allow(5000, S)); assert.ok(g.allow(16000, S));
  assert.ok(!new E.AlarmGate().allow(0, { ...S, autoAlert: false }));
});
test('stateColor mapping', () => {
  for (const [s, c] of [['alert', 'green'], ['possible_yawn', 'green'], ['possible_drowsy', 'green'], ['yawning', 'red'], ['drowsy', 'red'], ['eye_closure', 'red'], ['uncertain', 'gray'], ['no_face', 'gray']]) assert.equal(E.stateColor(s), c);
});
test('tracker keeps ids for moving faces and creates new ids', () => {
  const t = new E.Tracker(() => new E.Detector(S));
  const a = t.assign([{ x: .3, y: .5 }, { x: .7, y: .5 }], 0).tracks.map(x => x.id);
  const b = t.assign([{ x: .31, y: .5 }, { x: .69, y: .5 }], 100).tracks.map(x => x.id);
  assert.deepEqual(a, b); assert.equal(t.assign([{ x: .5, y: .1 }], 200).tracks[0].id, 3);
});
test('emotion softmax and uncertainty threshold', () => {
  const r = E.emotionResult([0, 5, 0, 0, 0, 0, 0, 0], .5); assert.equal(r.label, 'happy'); assert.ok(!r.uncertain);
  assert.ok(E.emotionResult([1, 1, 1, 1, 1, 1, 1, 1], .5).uncertain);
});
test('settings validation rejects out-of-range values', () => {
  const r = E.validateSettings({ earThresh: 5, volume: .3 }); assert.equal(r.errors.length, 1); assert.equal(r.settings.volume, .3); assert.equal(r.settings.earThresh, S.earThresh);
});
test('CSV escaping', () => { assert.equal(E.toCSV(['a', 'b'], [{ a: 'x,y', b: 'q"r' }]), 'a,b\r\n"x,y","q""r"'); });

test('closure longer than a blink but shorter than the limit is "possible drowsiness", no event', () => {
  const d = new E.Detector(S); const a = run(d, 600, { ear: .1, mar: .1 });
  assert.equal(a.r.state, 'possible_drowsy'); assert.equal(a.ev.length, 0);
});
test('smoothing: one noisy frame does not close the eyes or open the mouth', () => {
  const d = new E.Detector({ ...S, smooth: .3 }); run(d, 500, { ear: .3, mar: .1 });
  const r = d.update(600, { ear: .0, mar: 1.5 }); assert.ok(r.ear > .2 && r.mar < .6);
});
test('alarm gate reports remaining cooldown', () => {
  const g = new E.AlarmGate(); g.allow(1000, S); assert.equal(g.remaining(6000, S), 10000);
});
test('pipeline: landmarks -> MAR -> detector -> one yawn event -> one alert, then cooldown blocks the next', () => {
  const det = new E.Detector({ ...E.DEFAULTS }), gate = new E.AlarmGate(); let alerts = 0, onsets = 0;
  for (let t = 0; t <= 4000; t += 33) { // 4 s of a wide-open mouth, eyes open (smoothing on)
    const p = face(1, 1.4), r = det.update(t, { ear: E.ear(p), mar: E.mar(p) });
    for (const e of r.events) if (e.kind === 'onset' && e.type === 'yawn') { onsets++; if (gate.allow(t, E.DEFAULTS)) alerts++; }
  }
  assert.equal(onsets, 1); assert.equal(alerts, 1);
  det.update(4100, { ear: .5, mar: .1 }); for (let t = 4200; t <= 9000; t += 33) det.update(t, { ear: .5, mar: .1 }); // mouth closes: state resets
  assert.equal(det.state, 'alert');
  for (let t = 10000; t <= 13000; t += 33) for (const e of det.update(t, { ear: .5, mar: .9 }).events) if (e.kind === 'onset' && !gate.allow(t, E.DEFAULTS)) alerts += 0; else if (e.kind === 'onset') alerts++;
  assert.equal(alerts, 1, 'second yawn inside the 15 s cooldown is blocked');
});
