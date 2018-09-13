#!/usr/bin/env bash

#source /home1/irteam/.bashrc
#source /home1/irteam/.bash_profile
#source /home1/irteam/users/superneo/.bash_profile

total_models=5
cur_model=0
model_idcs=( 1 0 )  # 1 for baseline, 0 for test
model_labels=( "test" "baseline" )
ratios=( "00" "05" "10" "20" "50" )

runs_path="/home1/irteam/users/superneo/hclt/cnn-text-classification-tf/runs/"

while [ $cur_model -lt $total_models ]; do  # each model pair
    echo "<< model: $cur_model >>"

    for model_idx in "${model_idcs[@]}"; do  # baseline or test
        #echo $model_idx ${model_labels[$model_idx]}
        date_time=`date +'%Y%m%d_%H%M%S'`
        checkpoint_root=${model_labels[$model_idx]}_${cur_model}_${date_time}

        # training
        echo "./train.py $model_idx $checkpoint_root"
        CUDA_VISIBLE_DEVICES=7 python3 train.py $model_idx $checkpoint_root

        # test
        for ratio in "${ratios[@]}"; do
            pos_test_name="./data/nsmc/test/nsmc_pos_test_${ratio}.txt"
            neg_test_name="./data/nsmc/test/nsmc_neg_test_${ratio}.txt"
            echo "./eval.py -is_baseline $model_idx -checkpoint_dir ${runs_path}${checkpoint_root}/checkpoints -pos_test_file $pos_test_name -neg_test_file $neg_test_name -prediction_file prediction_${ratio}.csv -performance_file performance_${ratio}.txt"
            CUDA_VISIBLE_DEVICES=7 python3 eval.py -is_baseline $model_idx -checkpoint_dir "${runs_path}${checkpoint_root}/checkpoints" -pos_test_file $pos_test_name -neg_test_file $neg_test_name -prediction_file "prediction_${ratio}.csv" -performance_file "performance_${ratio}.txt"
        done

    done

    cur_model=$(($cur_model+1))
done

