# FaceSense AI (browser edition)

Same UI as the redesign, now with working detection. Everything runs **in your browser**. There is **no backend, database or sign-in** in this version (see "Not built").

## Run on Windows (VS Code terminal, PowerShell)
Requires Node.js 18+ (https://nodejs.org) and internet access (libraries load from jsDelivr).
```powershell
cd facesense
powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1
npm test
npm start
```
Open http://localhost:8080 in Chrome or Edge. Camera access works on `localhost` or HTTPS only (not on a plain `http://` network address).
No Node? `py -m http.server 8080` works as the web server; skip `npm test`.

## How to use
Sidebar: Dashboard, Driver Mode, Study Mode, Expressions, Emotion Distribution, Behavior Analytics, Settings.
Open **Driver Mode** → **Start monitoring** → allow the camera. Start also unlocks browser audio (browsers need a click); the Audio row shows its status. **Test siren** is a separate audio check. **Stop** saves the session. Emotion Distribution (pie/bar charts, filters) and Behavior Analytics (event charts, session history, CSV export, print) read the saved sessions.

Each face box shows: top label = expression (emoji, name, confidence; "Analyzing…" until the first result; "Uncertain" below your confidence limit), bottom tag = behavioural state. Box colour follows the behavioural state only.

## What runs where
| Feature | Where | Notes |
|---|---|---|
| Camera, face landmarks (MediaPipe Face Landmarker), eye/mouth measures | Browser | Video never leaves your device |
| Expression model (FER+ via onnxruntime-web) | Browser | Predicts 7 classes + "contempt" |
| Yawn, eye-closure, drowsiness logic, siren, Study boxes | Browser | `engine.js` |
| Sessions, settings, CSV | Browser `localStorage` | Per browser, auto-deleted after the retention days |
| Accounts, roles, database, API, audit log | **Not built** | |
| Identity verification and access policies | **Not built** | |

## Models and licenses
- `face_landmarker.task` – Google MediaPipe (Apache 2.0), from `storage.googleapis.com/mediapipe-models/...`
- `emotion-ferplus-8.onnx` – ONNX Model Zoo FER+ (MIT), 64x64 grayscale input, 8 outputs in the order neutral, happiness, surprise, sadness, anger, disgust, fear, contempt. Trained on FER+ lab-style images, so accuracy on webcams in poor light or at angles can be low.
- Libraries load from jsDelivr: `@mediapipe/tasks-vision@0.10.14`, `onnxruntime-web@1.18.0`.

## Detection logic (all thresholds editable in Settings)
- EAR (eye aspect ratio) from 6 landmarks per eye; closed below the threshold. Closures shorter than the prolonged-closure time are blinks and make no event.
- MAR (inner-lip height / mouth width); a yawn event needs the mouth open above the threshold for the sustained-yawn time. One event per continuous yawn.
- Drowsiness: share of eyes-closed frames over a 30 s window (PERCLOS-style, with hysteresis). Needs at least 15 s of data.
- Siren: fires on the *start* of a yawn, long-closure or drowsiness event in Driver mode, respects the cooldown, stops when the event ends or after 5 s.
- Study boxes: green = alert/normal, red = drowsy/prolonged closure/sustained yawn, gray = no face, face too small, or unavailable. Face numbers are temporary session tracks (nearest-centroid), not identities.
- Expression predictions run on the largest face only, about 3 times per second.

## Tests
`npm test` runs 11 automated tests of `engine.js` (EAR/MAR geometry, blink vs closure, yawn, PERCLOS drowsiness, no-face/uncertain, siren cooldown, box colours, tracker, softmax/threshold, settings validation, CSV).
The automated tests use synthetic landmark values. They do **not** test real webcam accuracy.

## Manual checklist (do these on your machine)
1. Start → camera light on; Stop → light off.
2. Model status shows "ready" for both models. If "UNAVAILABLE", run the download script and reload.
3. Smile / look neutral → Expressions page prediction changes; cover your face → "No face yet".
4. Close eyes for over 1 s → red box, event listed; blink → no event.
5. Open mouth wide for over 1.5 s → yawn event; short open → none.
6. Driver Mode open, hold your mouth wide open for 2 s → "Yawning detected", red box, banner with the reason, siren plays by itself; a second yawn inside the cooldown shows "No siren: cooldown" in the diagnostics.
7. Stop → session in Behavior Analytics and Emotion Distribution; export both CSVs; delete a session.
8. Two people in view (with consent) → two numbered boxes.

## Why the siren used to stay silent (fixed)
1. Alerts only fired if the session was started *while Driver mode was open*. Now they fire whenever Driver Mode is the open page.
2. Browser audio was only unlocked by a separate button, and failure was silent. Start now unlocks audio inside the click, and every alert records `siren: playing / audio not enabled or blocked / disabled in Settings`.
3. The cooldown was spent even when no sound played; the reason is now shown ("No siren: cooldown / Driver Mode not open / automatic alerts off").
4. Default mouth-open threshold lowered from 0.6 to 0.5 (MAR); if you saved Settings earlier, your old value is kept, so lower it there.
The **Development diagnostics** panel in Driver Mode shows MAR/EAR vs thresholds, open-event timers, audio state, cooldown and the last block reason. Remove the `<details>` block in `index.html` when you no longer need it.
