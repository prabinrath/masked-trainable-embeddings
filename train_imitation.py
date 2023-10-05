import argparse
from rl_bench.torch_data import load_data, ReverseTrajDataset
import numpy as np

FRANKA_JOINT_LIMITS = np.asarray(
    [
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    ],
    dtype=np.float32,
).T


def main():
    required_data_keys = [
        "front_rgb",
        "left_shoulder_rgb",
        "right_shoulder_rgb",
        "wrist_rgb",
        "joint_positions",
        "gripper_open",
    ]
    train_loader, val_dataset = load_data(
        dataset_dir="/home/local/ASUAD/opatil3/datasets/sim_open_close",
        required_data_keys=required_data_keys,
        task_filter_key=ReverseTrajDataset.task_filter_map["box"],
        chunk_size=100,
        norm_bound=FRANKA_JOINT_LIMITS,
    )

    for train_data in train_loader:
        pass


main()
