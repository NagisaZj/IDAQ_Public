#!/bin/bash
   # Script to reproduce results
 mkdir outlogs_offline &> /dev/null
 declare -a tasks=( "3s_vs_5z" )
 ## "3s_vs_5z" "2s3z" "2s_vs_1sc" "3s5z" "5m_vs_6m" "10m_vs_11m" "1c3s5z" "3c7z" "3h_vs_4z"
 declare -a buffer_ids=( "3" )
 ## "3" "5" "7"
 declare -a algos=( "dmaq" "vdn" "qmix" "qtran" )
 ## "dmaq" "vdn" "qmix" "qtran"
 declare -a seeds=( "0" "1" )
 n=0
 gpunum=8
 for task in "${tasks[@]}"
 do
 for buffer_id in "${buffer_ids[@]}"
 do
 for algo in "${algos[@]}"
 do
 for seed in "${seeds[@]}"
 do
 if [ ${algo} == 'qplex' ]
 then
 OMP_NUM_THREADS=16 KMP_AFFINITY="compact,granularity\=fine" CUDA_VISIBLE_DEVICES=${n} nohup python3 main.py \
 --config=dmaq_sc2 --env-config=sc2 with env_args.map_name=${task} \
 env_args.seed=1 \
 local_results_path='../../../tmp_DD/sc2_'${task}'/results/' \
 save_model=True \
 use_tensorboard=True \
 save_model_interval=200000 \
 t_max=2100000 \
 is_batch_rl=True \
 load_buffer_id=${buffer_id} \
 mac="mmdp_mac" \
 >& outlogs_offline/${task}_${buffer_id}_${algo}_${seed}.txt &
 else
 OMP_NUM_THREADS=16 KMP_AFFINITY="compact,granularity\=fine" CUDA_VISIBLE_DEVICES=${n} nohup python3 main.py \
 --config=${algo} --env-config=sc2 with env_args.map_name=${task} \
 env_args.seed=${seed} \
 local_results_path='../../../tmp_DD/sc2_'${task}'/results/' \
 save_model=True \
 use_tensorboard=True \
 save_model_interval=200000 \
 t_max=2100000 \
 is_batch_rl=True \
 load_buffer_id=${buffer_id} \
 mac="mmdp_mac" \
 >& outlogs_offline/${task}_${buffer_id}_${algo}_${seed}.txt &
 fi
 echo "task: ${task}, buffer_ids: ${buffer_id}, algo: ${algo}, seed: ${seed}, GPU: $n"
 n=$((($n+1) % ${gpunum}))
 sleep 5
 done
 done
 done
 done