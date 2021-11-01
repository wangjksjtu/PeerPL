# repeat exps with five random seeds

# Peer Reward
i=0
for log_dir in logs_01 logs_02 logs_03 logs_04 logs_05
do
for alpha in 0.1 0.2 0.3 0.4 0.5
do
if [ $(($i%8)) -eq 0 ]
then
    i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward peer --alpha $alpha)&
i=$(($i+1))

if [ $(($i%8)) -eq 0 ]
then
    i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward peer --alpha $alpha)&
i=$(($i+1))

if [ $(($i%8)) -eq 0 ]
then
    i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward peer --alpha $alpha)&
i=$(($i+1))

if [ $(($i%8)) -eq 0 ]
then
    i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward peer --alpha $alpha)&
i=$(($i+1))
done
sleep 1200
done

# Baselines
for log_dir in logs_01 logs_02 logs_03 logs_04 logs_05
do
# True Reward
(export CUDA_VISIBLE_DEVICES=0 && python dqn_cartpole.py --log_dir $log_dir)&

# Noisy Reward
(export CUDA_VISIBLE_DEVICES=0 && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward noisy)&
(export CUDA_VISIBLE_DEVICES=1 && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward noisy)&
(export CUDA_VISIBLE_DEVICES=2 && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward noisy)&
(export CUDA_VISIBLE_DEVICES=3 && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward noisy)&

# Surrogate Reward (Wang et al., 2020)
(export CUDA_VISIBLE_DEVICES=4 && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward surrogate)&
(export CUDA_VISIBLE_DEVICES=5 && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward surrogate)&
(export CUDA_VISIBLE_DEVICES=6 && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward surrogate)&
(export CUDA_VISIBLE_DEVICES=7 && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward surrogate)&
done
