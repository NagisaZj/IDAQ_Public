#!/bin/bash
   # Script to reproduce results

 Foldername="0619_offline_meta_rl"
 mkdir out_logs/${Foldername} &> /dev/null
 declare -a tasks=( "CPEARL-sparse-point-robot" )
 declare -a algos=( "cpearl" )
 ##
 declare -a seeds=( "1" )
 declare -a datadirs=( "sparse-point-robot-0.4" )
 # "sparse-point-robot-0.0" "sparse-point-robot-0.4" "sparse-point-robot-0.7"
 declare -a is_sparses=( "0" "1" )
 n=0
 gpunum=8
 for task in "${tasks[@]}"
 do
 for algo in "${algos[@]}"
 do
 for seed in "${seeds[@]}"
 do
 for datadir in "${datadirs[@]}"
 do
 for is_sparse in "${is_sparses[@]}"
 do
 OMP_NUM_THREADS=16 KMP_AFFINITY="compact,granularity\=fine" nohup python launch_experiment_${algo}.py \
 ./configs/${task}.json \
 ./data/${datadir} \
 --gpu=${n} \
 --is_sparse_reward=${is_sparse} \
 >& out_logs/${Foldername}/${task}_${algo}_${seed}_${datadir}_${is_sparse}.txt &
 echo "task: ${task}, algo: ${algo}, seed: ${seed}, datadir: ${datadir}, is_sparses: ${is_sparse}, GPU: $n"
 n=$[($n+1) % ${gpunum}]
 sleep 10
 done
 done
 done
 done
 done
