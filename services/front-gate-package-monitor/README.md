# Front gate package monitor

Lightweight, non-AI package-presence detection for the fixed front-gate camera.
It compares a configured delivery-zone ROI with an operator-captured empty
baseline, then requires a change to persist across multiple samples.

## Behavior

- Captures one 960px-wide sub-stream frame every 120 seconds normally.
- A trusted HA camera/doorbell event starts a 90-second burst at one frame
  every 10 seconds, allowing the existing three-sample confirmation to finish
  promptly after the visitor leaves.
- Analyzes only the upper parcel-shelf ROI, excluding the wall, lower shelf and
  sunlit paving.
- Compensates for global brightness changes.
- Uses morphological filtering and minimum contour area to reject noise.
- Requires one sufficiently large continuous changed region, so scattered
  moving shadows do not add up to a parcel alert.
- Requires three consecutive samples to enter or clear the package state.
- Keeps multiple empty-scene baselines for different daylight conditions.
- Adds a new baseline every 30 minutes only while the scene is confidently
  clear, so gradual lighting changes are followed without learning a detected
  object as background.
- Exposes read-only status and annotated snapshots over HTTP.

This service detects a persistent new object. It does not claim to classify the
object as a parcel, food delivery, person, animal, or plant.

## HTTP

- `GET /api/status`
- `POST /api/check` from the configured trusted HA host
- `GET /snapshot.jpg`
- `GET /baseline.jpg`
- `POST /api/calibrate` with `X-Calibration-Token`

## Calibration

Only calibrate when the delivery area is empty:

```sh
/home/guosq/workspace/stanley-rasp2/services/front-gate-package-monitor/calibrate.sh
```

Credentials and the calibration token live only in
`~/.config/front-gate-package-monitor.env`; the file must not be committed.
