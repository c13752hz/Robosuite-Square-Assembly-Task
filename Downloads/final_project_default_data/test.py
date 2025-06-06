'''

DO NOT MODIFY THIS FILE

'''

import numpy as np
import robosuite as suite
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from dmp_policy import DMPPolicyWithPID
from robosuite.utils.placement_samplers import UniformRandomSampler

placement_initializer = UniformRandomSampler(
    name="FixedOriSampler",
    mujoco_objects=None,            
    x_range=[-0.115, -0.11],       
    y_range=[0.05, 0.225],
    rotation=np.pi,
    rotation_axis="z",
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=False,
    reference_pos=(0,0,0.82),
    z_offset=0.02,
)

# create environment instance
env = suite.make(
    env_name="NutAssemblySquare", 
    robots="Panda", 
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    # placement_initializer=placement_initializer,
    ignore_done=True  
)

# piazza 
# env = suite.make(
#     env_name="NutAssemblySquare", 
#     robots="Panda", 
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     ignore_done=True,
#     camera_names=["agentview"],
# )

num_episodes = 100
max_steps_per_episode = 2500
success_rate = 0
steps_to_success = []
# reset the environment
for _ in range(num_episodes):
    obs = env.reset()
    policy = DMPPolicyWithPID(obs['SquareNut_pos'], obs['SquareNut_quat']) 
    step_count = None
    for t in range(max_steps_per_episode):
        action = policy.get_action(obs['robot0_eef_pos'], obs['robot0_eef_quat'])
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        if reward == 1.0:
            step_count = t + 1
            success_rate += 1
            break
    steps_to_success.append(step_count)  # None if failed


success_rate /= float(num_episodes)
print('success rate:', success_rate)

success_flags = [s is not None for s in steps_to_success]
total_successes = sum(success_flags)
total_failures = num_episodes - total_successes

# Filter out None (failures) to get only actual step counts
succeeded_steps = np.array([s for s in steps_to_success if s is not None])

plt.figure(figsize=(4, 3))
labels = ["Success", "Failure"]
counts = [total_successes, total_failures]
plt.bar(labels, counts, edgecolor='black')
plt.title("Episode Outcomes (100 Runs)")
plt.ylabel("Number of Episodes")
plt.tight_layout()
plt.savefig("episode_outcomes.png", dpi=200)
plt.close()

succeeded_steps = np.array([s for s in steps_to_success if s is not None])

if len(succeeded_steps) > 0:
    plt.figure(figsize=(5, 3))
    bins = np.arange(0, 2501, 500)  # buckets: 0–500, 500–1000, ..., 2000–2500
    plt.hist(succeeded_steps, bins=bins, edgecolor='black')
    plt.title("Steps to Success")
    plt.xlabel("Steps Taken Until Success")
    plt.ylabel("Number of Episodes")
    plt.xticks(bins)
    plt.tight_layout()
    plt.savefig("steps_to_success_hist.png", dpi=200)
    plt.close()
    # plt.show()
else:
    print("No successful episodes to plot histogram.")