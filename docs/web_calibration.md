# Web UI Calibration Workflow

This document describes the browser-based capture flow in `capture_handeye.py`.
It is meant for a robot host reached over SSH while you operate from a laptop
browser on the same network.

## 1. Start the Server

```bash
python capture_handeye.py \
  --mode eye-in-hand \
  --stream-debug \
  --stream-fk \
  --headless \
  --enable-arm-waypoints \
  --arm-network-interface <YOUR_DDS_INTERFACE> \
  --fk-network-interface <YOUR_DDS_INTERFACE> \
  --hand-frame right_wrist_yaw_link \
  --camera-mount wrist \
  --color-only \
  --cam-serial <YOUR_CAMERA_SERIAL> \
  --fk-urdf <PATH_TO_URDF> \
  --stream-host 0.0.0.0 \
  --stream-port 8080
```

Open:

```text
http://<robot-host-ip>:8080/
```

The process listens on all interfaces; restrict access with firewall rules if
needed.

## 2. Web Buttons

### Calibration

| Control | Description |
|---------|-------------|
| Save / Space | Save image + FK pose when chessboard is detected |
| Solve / S | Solve hand-eye transform from saved samples |
| Quit / Q | Stop the program |

### Arm Waypoints (`--enable-arm-waypoints`)

| Control | Description |
|---------|-------------|
| Save Current | Store current right-arm joints as a waypoint |
| Prev / Next | Select waypoint |
| Move | Smooth move to selected waypoint |
| Random Right Arm | Small random right-arm perturbation |
| Hold Current | Hold present posture |
| Arm Default | Release `arm_sdk` and request arm release |
| Release Arm SDK | Return arm control to internal controller |

Right-arm joint sliders (Δ / Go / Rand) send per-joint commands. Waist and
left-arm joints are forwarded as hold values so the upper-body command vector
stays complete.

## 3. Suggested Capture Sequence

1. Confirm e-stop and clearance around the arm.
2. Verify video and `fk` JSON update in the browser.
3. Save a home waypoint (`Save Current`).
4. Move the arm to another safe pose (teach, preset, or waypoint Move).
5. Wait for stable chessboard detection, then **Save**.
6. Repeat until you have 12–20 diverse poses.
7. Click **Solve**.
8. Release arm SDK when finished.

If only one waypoint exists, **Move** will not change posture.

## 4. Random Motion Limits

Default shoulder/elbow random delta: `0.08 rad`. Wrist: `0.18 rad`. Override:

```bash
--arm-random-shoulder-elbow-max-delta-rad 0.04 \
--arm-random-wrist-max-delta-rad 0.18 \
--arm-limit-margin-rad 0.03
```

## 5. Waypoint File

Default path per session:

```text
data/<session>/<camera_name>/arm_waypoints.json
```

Custom file:

```bash
--arm-waypoints-json data/my_waypoints.json
```

Example entry:

```json
{
  "name": "pose_a",
  "joints": {
    "right_shoulder_pitch": 0.29,
    "right_shoulder_roll": -0.22,
    "right_shoulder_yaw": 0.04,
    "right_elbow": 0.98,
    "right_wrist_roll": -0.10,
    "right_wrist_pitch": 0.01,
    "right_wrist_yaw": 0.02
  }
}
```

## 6. Safety

- `--enable-arm-waypoints` publishes to `rt/arm_sdk`. Test with small motions first.
- `Release Arm SDK` may allow internal control to return arms to a default pose.
- For a stable photo pose, move first, then Save — do not release before capture.
- By default, Move/Random auto-release SDK after settling; use
  `--no-arm-release-after-move` to hold until manual release.
- No leg/locomotion controls are exposed in the web UI.

## 7. H2 Upper-Arm Sweep

`random_upper_arm_sweep.py` is a second process. It heartbeats to this page
and receives the sweep buttons. Point `--web-url` at the same host and port
as `--stream-port`.

| Control | Description |
|---------|-------------|
| 下一个 n | Accept the current pose and move to the next sampled pose |
| 再抽 s | Reject the pose and sample another one |
| 结束 q | Stop the sweep |
| X/Y/Z | Translate in the torso frame. Z is up |
| R/P/Y | Rotate in the camera frame |
| 步长 /2, ×2 | Change the jog step |
| 跟随实测 | Reset the IK seed from the measured joints |

The sweep process is what publishes `rt/arm_sdk` for these buttons. Do not
also enable Arm Waypoints on the capture page at the same time.

Without `--confirm-robot-motion`, the sweep script only writes a YAML file.
`--mode dwell` waits at each pose. `--mode multi` runs the whole list without
waiting. Default sampling moves the shoulder and elbow only; add
`--include-wrist` when the wrist must move.

Gravity feedforward is on by default. Set `--wrist-payload-kg` from the
actual gripper and camera mass, or pass `--no-gravity` to send zero torque.
