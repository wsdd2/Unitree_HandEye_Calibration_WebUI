# FK URDF 测试样例数据

本目录给 `robot_kinematics/joint_to_pose/fk_urdf.py` 提供可直接运行的 Unitree G1 29DoF FK 样例数据。

## 文件说明

- `g1_fk_zero_joints.json`：零位姿样例，`joints` 为空，所有 active joint 由脚本默认填 `0.0`。
- `g1_fk_zero_expected.json`：零位姿对应的期望 FK 输出。
- `g1_fk_arm_pose_joints.json`：腰部和双臂非零关节样例，腿部关节缺省为 `0.0`。
- `g1_fk_arm_pose_expected.json`：非零手臂姿态对应的期望 FK 输出。

两个 `*_joints.json` 都保留了 `urdf`、`base_link`、`targets` 等说明字段；`fk_urdf.py --joints` 实际读取的是其中的 `joints` 字段。

## 运行示例

在本仓库根目录下执行：

```powershell
python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --base-link pelvis `
  --target d435_link,left_wrist_yaw_link,right_wrist_yaw_link `
  --joints .\robot_kinematics\example\g1_fk_zero_joints.json `
  --pretty
```

非零手臂姿态：

```powershell
python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --base-link pelvis `
  --target d435_link,left_wrist_yaw_link,right_wrist_yaw_link `
  --joints .\robot_kinematics\example\g1_fk_arm_pose_joints.json `
  --pretty
```

## 重新生成期望输出

如果 `fk_urdf.py` 的数值逻辑或 URDF 文件更新，可以用下面命令重新生成期望结果：

```powershell
python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --base-link pelvis `
  --target d435_link,left_wrist_yaw_link,right_wrist_yaw_link `
  --joints .\robot_kinematics\example\g1_fk_zero_joints.json `
  --pretty > .\robot_kinematics\example\g1_fk_zero_expected.json

python .\robot_kinematics\joint_to_pose\fk_urdf.py `
  --urdf .\unitree_ros\robots\g1_description\g1_29dof_rev_1_0.urdf `
  --base-link pelvis `
  --target d435_link,left_wrist_yaw_link,right_wrist_yaw_link `
  --joints .\robot_kinematics\example\g1_fk_arm_pose_joints.json `
  --pretty > .\robot_kinematics\example\g1_fk_arm_pose_expected.json
```
