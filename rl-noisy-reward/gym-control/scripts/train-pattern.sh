# seven dynamic schedules for peer penalty
i=0
for log_dir in logs_01 logs_02 logs_03 logs_04 logs_05
do
for pattern in 0 1 2 3 4 5 6 7
do

if [ $(($i%8)) -eq 0 ]
then
i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward peer --pattern $pattern)&
i=$(($i+1))

if [ $(($i%8)) -eq 0 ]
then
i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward peer --pattern $pattern)&
i=$(($i+1))

if [ $(($i%8)) -eq 0 ]
then
i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward peer --pattern $pattern)&
i=$(($i+1))

if [ $(($i%8)) -eq 0 ]
then
i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python dqn_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward peer --pattern $pattern)&
i=$(($i+1))
done
sleep 1200
done

for log_dir in logs_01 logs_02 logs_03 logs_04 logs_05
do
for pattern in 0 1 2 3 4 5 6 7
do

if [ $(($i%8)) -eq 0 ]
then
i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python duel_dqn_cartpole.py --log_dir $log_dir --error_positive 0.1 --reward peer --pattern $pattern)&
i=$(($i+1))

if [ $(($i%8)) -eq 0 ]
then
i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python duel_dqn_cartpole.py --log_dir $log_dir --error_positive 0.2 --reward peer --pattern $pattern)&
i=$(($i+1))

if [ $(($i%8)) -eq 0 ]
then
i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python duel_dqn_cartpole.py --log_dir $log_dir --error_positive 0.3 --reward peer --pattern $pattern)&
i=$(($i+1))

if [ $(($i%8)) -eq 0 ]
then
i=$(($i+1))
fi
(export CUDA_VISIBLE_DEVICES=$(($i%8)) && python duel_dqn_cartpole.py --log_dir $log_dir --error_positive 0.4 --reward peer --pattern $pattern)&
i=$(($i+1))
done
sleep 1200
done

