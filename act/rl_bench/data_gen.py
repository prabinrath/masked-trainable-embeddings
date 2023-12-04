import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
import pickle
import time
import lzma
import os


class ReverseTrajGen:
    reverse_task_pairs = {
        "open_close": [
            (OpenBox, CloseBox),
            # (OpenDoor, CloseDoor),
            # (OpenDrawer, CloseDrawer),
            # (OpenFridge, CloseFridge),
            # (OpenGrill, CloseGrill),
            # (OpenJar, CloseJar),
            # (OpenMicrowave, CloseMicrowave),
            # (ToiletSeatUp, ToiletSeatDown)
        ],
        "on_off": [
            (LampOn, LampOff),
            (MeatOnGrill, MeatOffGrill),
            (PutToiletRollOnStand, TakeToiletRollOffStand),
        ],
        "in_out": [
            (LightBulbIn, LightBulbOut),
            (InsertUsbInComputer, TakeUsbOutOfComputer),
            (PlugChargerInPowerSupply, UnplugCharger),
            (PutUmbrellaInUmbrellaStand, TakeUmbrellaOutOfUmbrellaStand),
            (PutTrayInOven, TakeTrayOutOfOven),
            (PutShoesInBox, TakeShoesOutOfBox),
            (PutPlateInColoredDishRack, TakePlateOffColoredDishRack),
            (PutMoneyInSafe, TakeMoneyOutSafe),
        ],
    }

    def __init__(self, robot="panda", headless=True):
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        self.env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
            ),
            obs_config=ObservationConfig(),
            robot_setup=robot,
            headless=headless,
        )
        self.env.launch()

    def get_demo(self, Task):
        demo_task = self.env.get_task(Task)
        print(demo_task.get_name())
        demo_task.sample_variation()
        start = time.perf_counter()
        demo = demo_task.get_demos(1, live_demos=True)[0]
        demo_dict = dict(
            front_rgb=[],
            front_mask=[],
            front_depth=[],
            front_point_cloud=[],
            left_shoulder_rgb=[],
            left_shoulder_mask=[],
            left_shoulder_depth=[],
            left_shoulder_point_cloud=[],
            right_shoulder_rgb=[],
            right_shoulder_mask=[],
            right_shoulder_depth=[],
            right_shoulder_point_cloud=[],
            overhead_rgb=[],
            overhead_mask=[],
            overhead_depth=[],
            overhead_point_cloud=[],
            wrist_rgb=[],
            wrist_mask=[],
            wrist_depth=[],
            wrist_point_cloud=[],
            joint_positions=[],
            joint_velocities=[],
            gripper_pose=[],
            gripper_open=[],
        )
        for obs in demo._observations:
            demo_dict["front_rgb"].append(obs.front_rgb)
            demo_dict["front_mask"].append(obs.front_mask)
            demo_dict["front_depth"].append(obs.front_depth)
            # demo_dict['front_point_cloud'].append(obs.front_point_cloud)
            demo_dict["left_shoulder_rgb"].append(obs.left_shoulder_rgb)
            demo_dict["left_shoulder_mask"].append(obs.left_shoulder_mask)
            demo_dict["left_shoulder_depth"].append(obs.left_shoulder_depth)
            # demo_dict['left_shoulder_point_cloud'].append(obs.left_shoulder_point_cloud)
            demo_dict["right_shoulder_rgb"].append(obs.right_shoulder_rgb)
            demo_dict["right_shoulder_mask"].append(obs.right_shoulder_mask)
            demo_dict["right_shoulder_depth"].append(obs.right_shoulder_depth)
            # demo_dict['right_shoulder_point_cloud'].append(obs.right_shoulder_point_cloud)
            demo_dict["overhead_rgb"].append(obs.overhead_rgb)
            demo_dict["overhead_mask"].append(obs.overhead_mask)
            demo_dict["overhead_depth"].append(obs.overhead_depth)
            # demo_dict['overhead_point_cloud'].append(obs.overhead_point_cloud)
            demo_dict["wrist_rgb"].append(obs.wrist_rgb)
            demo_dict["wrist_mask"].append(obs.wrist_mask)
            demo_dict["wrist_depth"].append(obs.wrist_depth)
            # demo_dict['wrist_point_cloud'].append(obs.wrist_point_cloud)
            demo_dict["joint_positions"].append(obs.joint_positions)
            demo_dict["joint_velocities"].append(obs.joint_velocities)
            demo_dict["gripper_pose"].append(obs.gripper_pose)
            demo_dict["gripper_open"].append(obs.gripper_open)

        print(f"demo generated in: {round(time.perf_counter()-start,3)}s")
        return demo_dict

    def generate_data(self, category, dataset_path, num_demos_per_task=100):
        assert category in self.reverse_task_pairs
        task_pairs = self.reverse_task_pairs[category]
        for i, (Forward, Backward) in enumerate(task_pairs):
            for j in range(num_demos_per_task):
                with open(
                    os.path.join(
                        dataset_path,
                        f"sim_{category}_1k/demo_forward_{(j+1)*10+(i+1)}.pickle",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(self.get_demo(Forward), f)
                with open(
                    os.path.join(
                        dataset_path,
                        f"sim_{category}_1k/demo_backward_{(j+1)*10+(i+1)}.pickle",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(self.get_demo(Backward), f)


traj_gen = ReverseTrajGen(headless=True)
traj_gen.generate_data(
    category="open_close",
    dataset_path="/home/local/ASUAD/opatil3/datasets/",
    num_demos_per_task=1000,
)
