# Weakly Supervised Imitation Learning and Co-training

This repo contains the codes you need to replicate our experiments of imitation learning tasks and co-training for reinforcement learning tasks. The framework is based on `stable-baselines` and `rl-baselines-zoo`, thanks to the original authors！

## Dependencies

See `requirements.txt`

## Imitation Learning from Imperfect Expert (Atari)

1. First, we need to train an imperfect expert model. Open `rl-baselines-zoo`, run `python3 train.py --env {ENV} --save-interval 1000000`, where `ENV` should be the id of the environment you want to use, like `PongNoFrameskip-v4`. This will train a ppo model on Pong for `10^7` steps and save the model every `10^6` steps.
2. Copy the folder containing generated expert models (like `rl-baseines-zoo/logs/ppo2/PongNoFrameskip-v4_1/`) to `logs/{ENV}/baseline/`.
3. Now we can generate the expert trajectories. `python -m stable_baselines.gail.dataset.record_expert logs/{ENV}/baseline/{STEP}_steps /expert.npz --env {ENV} --note {STEP}`, where `{STEP}` is the step of the trained model you want to use, like `2000000`. This will generate 100 trajectories and store them in `logs/{ENV}/baseline/{STEP}/`.
4. Finallly, we can leverage these trajectories as weak supervision signals to train our agent. `python3 -m stable_baselines.ppo2.peer_behavior_clone logs/{ENV}/baseline/{STEP} --env {ENV} --seed {SEED} --policy {POLICY} --peer {XI} --note {NOTE} --val-interval {VAL_INTERVAL} --num-epochs {NUM_EPOCHS}`. Here, `{SEED}` is the global random seed, `{POLICY}` is the network type we want to use, chosen from `CnnPolicy` and `MlpPolicy`, `{XI}` is the coefficient of the peer term, `{NOTE}` is the experiment id set by the user, `{VAL_INTERVAL}` and `{NUM_EPOCHS}` control the evaluate frequency and the total training epochs.

## Co-training for Policy Learning (Gym & Atari)

For co-training tasks, we just need to run `python3 -m stable_baselines.ppo2.copier --env {ENV} --policy {POLICY} --seed {SEED} --peer {XI} --note {NOTE} --start-episode {START_EPISODE} --end-episode {END_EPISODE} --decay-rate {DECAY_TYPE} --total-steps {TOTAL_STEPS} (--individual)`.New arguments are explained as following: `START_EPISODE` and `END_EPISODE` decide the episode interval where peer term is added, `{DECAY_TYPE}` decides the changing of `XI`, and `{TOTAL_TIMESTEPS}` is the total number of timesteps to train.