# Masked Trainable Embeddings (MTE)

![MTE_Poster](./mte.png)

### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``constants.py`` Constants shared across files
- ``utils.py`` Helper functions
- ``rl_bench`` Data generation and loader

### Installation

    - Install ACT
        conda create -n aloha python=3.8.10
        conda activate aloha
        pip install torchvision
        pip install torch
        pip install pyquaternion
        pip install pyyaml
        pip install rospkg
        pip install pexpect
        pip install mujoco
        pip install dm_control
        pip install opencv-python
        pip install matplotlib
        pip install einops
        pip install packaging
        pip install h5py
        cd act/detr && pip install -e .

    - Install RLBench in the same env
        https://github.com/stepjam/RLBench


### Simulated experiments

- Add DATA_DIR in constants file: this is where the demonstrations are stored
- To add language conditioning, add ```--add_task_ind``` to the arguments for both training and eval
- To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
- To specify the the model checkpoints for rollout, use ```--ckpt_names``` in the argument followed by the checkpoint names
- To enable temporal ensembling, add flag ``--temporal_agg``.
- Videos will be saved to ``<ckpt_dir>`` for each rollout.
- You can also add ``--onscreen_render`` to see real-time rendering during evaluation.
