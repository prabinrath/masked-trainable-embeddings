import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ToiletSeatDown
import time

class Agent(object):
    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=ObservationConfig(),
    robot_setup='panda',
    headless=False)
env.launch()

agent = Agent(env.action_shape)

steps_per_task = 200
num_episodes = 10

task = env.get_task(ToiletSeatDown)
task.sample_variation()  # random variation

for e in range(num_episodes):
    print('Reset Episode')
    descriptions, obs = task.reset()
    print(descriptions)
    for i in range(steps_per_task):
        action = agent.act(obs)
        start = time.perf_counter()
        obs, reward, terminate = task.step(action)
        print(f'step time: {time.perf_counter()-start}')

# start = time.perf_counter()
# demos = task.get_demos(1, live_demos=True)
# print(f'demo gen time: {time.perf_counter()-start}')

print('Done')
env.shutdown()