import numpy as np
from collections import defaultdict
from dmp import DMP
from pid import PID
from load_data import reconstruct_from_npz
from scipy.spatial.transform import Rotation as R

class DMPPolicyWithPID:
    def __init__(self, square_pose, square_quat, demo_path='demos.npz', dt=0.01, n_bfs=50):
        self.dt = dt
        self.segment_dmps = []
        self.segment_targets = []
        self.segment_grasps = []
        self.segment_quats = []
        self.rot_seg0 = False
        
        self.hold_grip_counter = 0
        self.stall_counter = 0
        self.last_step_index = 0
        self.max_stall_steps = int(6.0 / self.dt) 


        demos = reconstruct_from_npz(demo_path)
        if demos is None:
            raise ValueError(f"Failed to load demo data from {demo_path}")

        print(f"Loaded {len(demos)} demonstrations.")

        for demo_id, demo in demos.items():
            if not all(key in demo for key in ['obs_robot0_eef_pos', 'obs_robot0_eef_quat', 'actions', 'obs_object']):
                print(f"Skipping {demo_id}: missing required fields.")
                continue

            ee_pos = demo['obs_robot0_eef_pos']
            ee_quat = demo['obs_robot0_eef_quat']
            ee_grasp = demo['actions'][:, -1:].astype(int)
            segments = self.detect_grasp_segments(ee_grasp)

            if not segments:
                print(f"Skipping {demo_id}: no grasp-based segments found.")
                continue

            demo_obj_pos = demo['obs_object'][0, :3]
            demo_obj_quat = demo['obs_object'][0, 3:7]
            R_demo = R.from_quat(demo_obj_quat)
            R_target = R.from_quat(square_quat)
            R_diff = R_target * R_demo.inv()
            offset = square_pose - demo_obj_pos

            for i, (start, end) in enumerate(segments):
                if end - start < 2:
                    print(f"Skipping short segment in {demo_id}: [{start}, {end}]")
                    continue

                segment_traj = ee_pos[start:end].copy().T  # shape (3, T)
                segment_quat = ee_quat[start:end]       

                if i == 0:
                    segment_traj -= demo_obj_pos[:, None]
                    segment_traj = R_diff.apply(segment_traj.T).T
                    ee_rot = R.from_quat(segment_quat)
                    rotated_quat = (R_diff * ee_rot).as_quat()
                    segment_traj += square_pose[:, None]
                else:
                    segment_traj = ee_pos[start:end].T  # shape (3, T)
                    segment_traj[2, :] += 0.09
                    segment_traj[0, :] += 0.015
                    rotated_quat = segment_quat


                dmp = DMP(n_dmps=3, n_bfs=n_bfs, dt=self.dt, y0=segment_traj[:, 0], goal=segment_traj[:, -1])
                dmp.imitate(segment_traj)

                self.segment_dmps.append(dmp)
                self.segment_targets.append(dmp.rollout())
                self.segment_grasps.append(int(np.round(ee_grasp[start][0])))
                self.segment_quats.append(rotated_quat)

        self.pids = [PID(kp=4.0, ki=0.0, kd=0.1, target=traj[0]) for traj in self.segment_targets]
        for pid in self.pids:
            pid.reset()

        self.current_segment = 0
        self.current_step = 0

    def detect_grasp_segments(self, grasp_flags: np.ndarray) -> list:
        grasp_flags = grasp_flags.squeeze()
        change_indices = np.where(np.diff(grasp_flags) != 0)[0] + 1
        segment_boundaries = [0] + change_indices.tolist() + [len(grasp_flags)]
        return [(segment_boundaries[i], segment_boundaries[i + 1]) for i in range(len(segment_boundaries) - 1)]

    def get_action(self, robot_eef_pos: np.ndarray, robot_eef_quat: np.ndarray) -> np.ndarray:
        if self.current_segment >= len(self.segment_targets):
            return np.zeros(7)

        target_traj = self.segment_targets[self.current_segment]
        target_quats = self.segment_quats[self.current_segment]
        grasp = self.segment_grasps[self.current_segment]

        step_idx = min(self.current_step, len(target_traj) - 1, len(target_quats) - 1)
        if step_idx > len(target_quats) - 1:
            desired_pos = target_traj[step_idx]
            desired_quat = target_quats[len(target_quats) - 1]
        else: 
            desired_pos = target_traj[step_idx]
            desired_quat = target_quats[step_idx]

        delta_pos = desired_pos - robot_eef_pos
        delta_pos = np.clip(delta_pos, -0.1, 0.1) 

        pos_error = np.linalg.norm(delta_pos)
        pos_done = pos_error < 0.024

        current_rot = R.from_quat(robot_eef_quat)
        target_rot = R.from_quat(desired_quat)
        delta_rotvec = (target_rot * current_rot.inv()).as_rotvec()
        delta_rotvec = np.clip(delta_rotvec, -0.05, 0.05)
        delta_err = np.linalg.norm(delta_rotvec)
        rot_done = delta_err < 0.013

        if pos_done and rot_done:
            print(self.current_segment)
            print("step completed")
            self.current_segment += 1
            self.current_step = 0
            if self.current_segment >= len(self.segment_targets):
                print("All segments done.")
                return np.zeros(7)
            return np.zeros(7)
        
        if self.current_segment == 1:
            if self.hold_grip_counter < 20:
                self.hold_grip_counter += 1
                print(f"Holding grip... ({self.hold_grip_counter}/20)")
                return [0,0,0,0,0,0,1] 


        if pos_error < 0.05 and self.current_step < len(target_traj) - 1:
            print(self.current_step)
            self.current_step += 1

        if self.current_step == self.last_step_index:
            self.stall_counter += 1
            if self.stall_counter >= self.max_stall_steps:
                print("Stuck too long, forcing large movement")
                print(pos_error)
                self.stall_counter = 0
                self.current_step += 1
                print(self.current_step)
                return [0,0,0,0,0,1,0]
        else:
            self.stall_counter = 0
            self.last_step_index = self.current_step

        action = np.zeros(7)
        action[:3] = delta_pos
        action[3:6] = delta_rotvec
        action[6] = grasp
        return action
