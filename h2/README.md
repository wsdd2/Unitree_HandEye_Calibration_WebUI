# H2 Hand-Eye Wrapper

Thin adapter around the core `capture_handeye.py` in the repository root. The
original calibration code is **not** modified; this folder only patches live FK
to match common H2 Cartesian conventions.

## Virtual End-Effector `R_ee`

Hardware H2 Cartesian scripts often report pose at:

```text
R_ee = right_wrist_yaw_joint frame + [0.05, 0, 0] meters
```

This represents the arm-to-gripper mounting point **without** Dex1-1 finger
kinematics. The wrapper exposes `fk.targets.R_ee` in the web JSON and uses
`R_ee` as `hand_frame` during capture.

## Waist Lock

To align with reduced-model FK used in many H2 control stacks, the wrapper
forces:

```text
waist_yaw_joint   = 0
waist_roll_joint  = 0
waist_pitch_joint = 0
```

Measured waist values are still published under
`fk.h2_virtual_ee.measured_waist_rad` for debugging.

## Run

```bash
cd h2
python3 capture_h2_handeye.py \
  --cam-serial <YOUR_CAMERA_SERIAL> \
  --fk-urdf <PATH_TO_H2.urdf> \
  --fk-network-interface <YOUR_DDS_INTERFACE> \
  --stream-port 8080
```

Or:

```bash
./run_h2_handeye.sh --cam-serial <YOUR_CAMERA_SERIAL>
```

Browser:

```text
http://<robot-host-ip>:8080/
```

All CLI flags supported by `capture_handeye.py` can be appended after the
wrapper command.

### Recommended board

```text
11 x 8 inner corners, 20 mm squares
```

Adjust with `--cols`, `--rows`, `--square-mm`.

## Expected Web FK JSON (excerpt)

```json
{
  "fk": {
    "base_link": "pelvis",
    "hand_frame": "R_ee",
    "targets": {
      "right_wrist_yaw_link": {},
      "R_ee": {}
    },
    "h2_virtual_ee": {
      "frame": "R_ee",
      "parent": "right_wrist_yaw_link",
      "offset_xyz_m": [0.05, 0.0, 0.0],
      "waist_locked": true
    }
  }
}
```

`fk.targets.R_ee.transform_matrix` is stored as the robot hand pose for each
sample.

## FK Comparison Tool

When your FK differs from another H2 toolchain, run the read-only comparator
(it subscribes to `rt/lowstate` only):

```bash
python3 compare_h2_fk.py --iface <YOUR_DDS_INTERFACE> --period 2.0 --lock-waist
```

If `--lock-waist` makes the error vanish, the mismatch is caused by waist
joints being included in one FK chain but locked in the other.

Optional hardware reference FK (Pinocchio) requires a local
`H2_joint_cartesian` checkout:

```bash
python3 compare_h2_fk.py \
  --iface <YOUR_DDS_INTERFACE> \
  --h2-root <PATH_TO_H2_joint_cartesian> \
  --lock-waist
```
