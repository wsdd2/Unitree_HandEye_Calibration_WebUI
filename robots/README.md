# Robot URDF Models

This repository does **not** ship Unitree robot URDF meshes. Provide your own
URDF files locally and pass the path via CLI flags such as `--fk-urdf`.

Suggested layout:

```text
robots/
  g1/
    g1_29dof_mode_15_with_dex1_1.urdf   # G1 with Dex1-1 (optional)
    g1_29dof_rev_1_0.urdf
  h2/
    H2.urdf                             # H2 without gripper fingers
```

Typical sources:

- Unitree open-source robot description packages
- Your on-robot `H2_joint_cartesian` / `unitree_ros` checkout

The FK helper under `robot_kinematics/` only needs a standard URDF; mesh files
are optional for pose computation.
