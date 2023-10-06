# latent-actions
 
- Add DATA_DIR in constants file- this is where the demonstrations are stored
cd act/
task name:sim_door_open/sim_door_close 
python3  imitate_episodes.py --task_name sim_door_close --ckpt_dir /home/local/ASUAD/<add_location>/act_door_close_100 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0

task name: sim_box_close/sim_box_open
python3  imitate_episodes.py --task_name sim_box_close --ckpt_dir /home/local/ASUAD/<add_location>/act_box_close_100 --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 128 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0
