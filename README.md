# robot-latent-actions

### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset

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
        pip install ipython
        cd act/detr && pip install -e .
    
    - Install RLBench in the same env
        https://github.com/stepjam/RLBench


### Simulated experiments

- Add DATA_DIR in constants file- this is where the demonstrations are stored
  
cd act/

**task name: sim_door_open/sim_door_close**  
python3  imitate_episodes.py --task_name sim_door_close --ckpt_dir /home/local/ASUAD/<add_location>/act_door_close_100 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0

**task name: sim_box_close/sim_box_open**   
python3  imitate_episodes.py --task_name sim_box_close --ckpt_dir /home/local/ASUAD/<add_location>/act_box_close_100 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0

To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.