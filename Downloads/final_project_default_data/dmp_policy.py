import numpy as np
from collections import defaultdict
from dmp import DMP
from pid import PID
from load_data import reconstruct_from_npz

class DMPPolicyWithPID:
    """
    A policy that follows a demonstrated path with DMPs and PID control.

    The demonstration is split into segments based on grasp toggles.  
    The first segment's endpoint is re-targeted to a new object pose.
    Subsequent segments replay the original DMP rollouts.

    Args:
        square_pos (np.ndarray): Target object position.
        demo_path (str): path to .npz file with demo data.
        dt (float): control timestep.
        n_bfs (int): number of basis functions per DMP.
    """
    def __init__(self, square_pos, demo_path='demos.npz', dt=0.01, n_bfs=50):
        self.dt = dt
        self.segment_dmps = []
        self.segment_targets = []
        self.segment_grasps = []

        demos = reconstruct_from_npz(demo_path)
        if demos is None:
            raise ValueError(f"Failed to load demo data from {demo_path}")

        print(f"Loaded {len(demos)} demonstrations.")

        for demo_id, demo in demos.items():
            if not all(key in demo for key in ['obs_robot0_eef_pos', 'actions', 'obs_object']):
                print(f"Skipping {demo_id}: missing required fields.")
                continue

            ee_pos = demo['obs_robot0_eef_pos']  # (T,3)
            ee_grasp = demo['actions'][:, -1:].astype(int)  # (T,1)
            segments = self.detect_grasp_segments(ee_grasp)

            if not segments:
                print(f"Skipping {demo_id}: no grasp-based segments found.")
                continue

            demo_obj_pos = demo['obs_object'][0, :3]
            offset = square_pos - demo_obj_pos

            for i, (start, end) in enumerate(segments):
                if end - start < 2:
                    print(f"Skipping short segment in {demo_id}: [{start}, {end}]")
                    continue

                segment_traj = ee_pos[start:end].T
                if i == 0:
                    segment_traj = segment_traj.copy()
                    segment_traj += offset[:, None]
                    segment_traj[0, :] -= 0.02 
                    segment_traj[1, :] += 0.045 
                    segment_traj[2, :] -= 0.008
                elif i == 1:
                    segment_traj = segment_traj.copy()
                    segment_traj[2, :] += 0.03
                    segment_traj[0, :] += 0.03

                dmp = DMP(n_dmps=3, n_bfs=n_bfs, dt=self.dt, y0=segment_traj[:, 0], goal=segment_traj[:, -1])
                dmp.imitate(segment_traj)
                self.segment_dmps.append(dmp)
                self.segment_targets.append(dmp.rollout())
                self.segment_grasps.append(int(np.round(ee_grasp[start][0])))

        self.pids = [PID(kp=2.0, ki=0.0, kd=0.0, target=traj[0]) for traj in self.segment_targets]
        for pid in self.pids:
            pid.reset()

        self.current_segment = 0
        self.current_step = 0

    def detect_grasp_segments(self, grasp_flags: np.ndarray) -> list:
        """
        Identify segments based on grasp toggles.

        Args:
            grasp_flags (np.ndarray): (T,1) array of grasp signals.

        Returns:
            List[Tuple[int,int]]: start and end indices per segment.
        """
        grasp_flags = grasp_flags.squeeze()
        change_indices = np.where(np.diff(grasp_flags) != 0)[0] + 1
        segment_boundaries = [0] + change_indices.tolist() + [len(grasp_flags)]
        return [(segment_boundaries[i], segment_boundaries[i + 1]) for i in range(len(segment_boundaries) - 1)]

    def get_action(self, robot_eef_pos: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos (np.ndarray): Current end-effector position [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,0,0,0,grasp].
        """
        if self.current_segment >= len(self.segment_targets):
            return np.zeros(7)

        target_traj = self.segment_targets[self.current_segment]
        if self.current_step >= target_traj.shape[0]:
            self.current_segment += 1
            self.current_step = 0
            if self.current_segment >= len(self.segment_targets):
                return np.zeros(7)
            target_traj = self.segment_targets[self.current_segment]

        desired_pos = target_traj[self.current_step]
        pid = self.pids[self.current_segment]
        pid.target = desired_pos  
        delta_pos = pid.update(robot_eef_pos, dt=self.dt)

        grasp = self.segment_grasps[self.current_segment]
        action = np.zeros(7)
        action[:3] = delta_pos
        action[-1] = grasp

        self.current_step += 1
        return action
