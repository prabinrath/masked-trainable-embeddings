import pathlib
from rlbench.tasks import OpenDoor, OpenBox, CloseBox, CloseDoor

### Task parameters
DATA_DIR = "/home/local/ASUAD/opatil3/datasets/"
SIM_TASK_CONFIGS = {
    "sim_box_open": {
        "rlbench_env": [OpenBox],
        "dataset_dir": DATA_DIR + "/sim_open_close",
        "num_episodes": 100,
        "episode_len": 250,
        "camera_names": [
            "front_rgb",
            "left_shoulder_rgb",
            "right_shoulder_rgb",
            "wrist_rgb",
        ],
    },
    "sim_box_close": {
        "rlbench_env": [CloseBox],
        "dataset_dir": DATA_DIR + "/sim_open_close",
        "num_episodes": 100,
        "episode_len": 250,
        "camera_names": [
            "front_rgb",
            "left_shoulder_rgb",
            "right_shoulder_rgb",
            "wrist_rgb",
        ],
    },
    "sim_door_open": {
        "rlbench_env": [OpenDoor],
        "dataset_dir": DATA_DIR + "/sim_open_close",
        "num_episodes": 100,
        "episode_len": 250,
        "camera_names": [
            "front_rgb",
            "left_shoulder_rgb",
            "right_shoulder_rgb",
            "wrist_rgb",
        ],
    },
    "sim_door_close": {
        "rlbench_env": [CloseDoor],
        "dataset_dir": DATA_DIR + "/sim_open_close",
        "num_episodes": 100,
        "episode_len": 250,
        "camera_names": [
            "front_rgb",
            "left_shoulder_rgb",
            "right_shoulder_rgb",
            "wrist_rgb",
        ],
    },
    # For model trained on both openbox and closebox demonstrations
    "sim_box": {
        "rlbench_env": [OpenBox, CloseBox],
        "dataset_dir": DATA_DIR + "/sim_open_close",
        "num_episodes": 200,
        "episode_len": 500,
        "camera_names": [
            "front_rgb",
            "left_shoulder_rgb",
            "right_shoulder_rgb",
            "wrist_rgb",
        ],
    },
    # For model trained on both opendoor and closedoor demonstrations
    "sim_door": {
        "rlbench_env": [OpenDoor, CloseDoor],
        "dataset_dir": DATA_DIR + "/sim_open_close",
        "num_episodes": 200,
        "episode_len": 500,
        "camera_names": [
            "front_rgb",
            "left_shoulder_rgb",
            "right_shoulder_rgb",
            "wrist_rgb",
        ],
    },
}
