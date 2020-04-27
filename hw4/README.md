Problem 1

python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n500_arch1x32 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1

data/mb_cheetah_n500_arch1x32_cheetah-cs285-v0_24-04-2020_18-15-39



python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n5_arch2x250 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 5 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1

data/mb_cheetah_n5_arch2x250_cheetah-cs285-v0_24-04-2020_19-08-04



python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n500_arch2x250 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 2 --size 250  --scalar_log_freq -1 --video_log_freq -1

data/mb_cheetah_n500_arch2x250_cheetah-cs285-v0_24-04-2020_19-10-37







1) See hw1 if you'd like to see installation instructions. You do NOT have to redo them.

##############################################
##############################################


2) Code:

-------------------------------------------

Files to look at, even though there are no explicit 'TODO' markings:
- scripts/run_hw4_mb.py

-------------------------------------------

Blanks to be filled in by using your code from hw1 are marked with 'TODO: GETTHIS from HW1'

The following files have these:
- infrastructure/rl_trainer.py
- infrastructure/utils.py
- policies/MLP_policy.py

Blanks to be filled in by using your code from hw2 are marked with 'TODO: GETTHIS from HW2'

- infrastructure/utils.py
- policies/MLP_policy.py

-------------------------------------------

Blanks to be filled in now (for this assignment) are marked with 'TODO'

The following files have these:
- critics/mb_agent.py
- models/ff_model.py
- policies/MPC_policy.py
- infrastructure/utils.py

##############################################
##############################################


3) Commands: 

Please refer to the PDF for the specific commands needed for different questions. 

##############################################


4) Visualize saved tensorboard event file:

$ cd cs285/data/<your_log_dir>
$ tensorboard --logdir .

Then, navigate to shown url to see scalar summaries as plots (in 'scalar' tab), as well as videos (in 'images' tab)