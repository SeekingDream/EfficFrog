


#TaskList="0"
#for task in $TaskList
#do
#  echo "$task"
##  CUDA_VISIBLE_DEVICES=4 python exp_train_clean_model.py --exp="$task"
##
##  CUDA_VISIBLE_DEVICES=4 python trigger_generation.py --exp="$task"
#  #
#  #  CUDA_VISIBLE_DEVICES=4 python exp_inject_backdoor.py --exp="$task" --type=0
#  CUDA_VISIBLE_DEVICES=4 python exp_inject_backdoor.py --exp="$task" --type=1
#
#  # CUDA_VISIBLE_DEVICES=4 python exp_inject_baseline.py --exp="$task" --baseline=0
#  # CUDA_VISIBLE_DEVICES=4 python exp_inject_baseline.py --exp="$task" --baseline=1
#done

python attack_wanet.py --exp=0 --device=1
python attack_sleep.py --exp=0 --device=1
python attack_issba.py --exp=0 --device=1





# CUDA_VISIBLE_DEVICES=1 python exp_inject_backdoor.py --exp=0



#CUDA_VISIBLE_DEVICES=0 python exp_inject_baseline.py --exp=3 --baseline 0
#CUDA_VISIBLE_DEVICES=0 python exp_inject_baseline.py --exp=3 --baseline 1




# CUDA_VISIBLE_DEVICES=2 python trigger_generation.py --exp=3