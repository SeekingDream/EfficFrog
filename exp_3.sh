#
#TaskList="2"
#for task in $TaskList
#do
#  echo "$task"
##  CUDA_VISIBLE_DEVICES=6 python exp_train_clean_model.py --exp="$task"
##
##  CUDA_VISIBLE_DEVICES=6 python trigger_generation.py --exp="$task"
##
##  CUDA_VISIBLE_DEVICES=6 python exp_inject_backdoor.py --exp="$task" --type=0
#    CUDA_VISIBLE_DEVICES=6 python exp_inject_backdoor.py --exp="$task" --type=1
##
##  CUDA_VISIBLE_DEVICES=6 python exp_inject_baseline.py --exp="$task" --baseline=0
##  CUDA_VISIBLE_DEVICES=6 python exp_inject_baseline.py --exp="$task" --baseline=1
#done


python attack_wanet.py --exp=2 --device=3
python attack_sleep.py --exp=2 --device=3
python attack_issba.py --exp=2 --device=3