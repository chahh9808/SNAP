#!/bin/bash
cd ../../
date=$1
data="IN" 
model="resnet50"
device="cuda"
workers="2"
test_corrupt=$2
eval_mode="continual"
seeds=(0 1 2)

LOG_DIR=logs/txtlogs/"$data"log/"$date"x/"$date""$test_corrupt"_"$data"log
# Check if the directory exists, if not create it
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

#IN
algs=( "sar" )
lr="1e-3"
iabn_k="2"
iabn_s="1"
maxage='1'

gpu_count=8
initial_gpu_id=0
gpu_id=$initial_gpu_id 

adaptrates=(0.01 0.03 0.05 0.1 0.3 0.5)
memtypes=('pb')
mem_sizes=(16)

adst_rmst_pairs_b=(
  "basic-RAND"
)

adst_rmst_pairs_i=(
  # "all-RAND"
  # "low_entr-RAND"
  "high_conf-WASS_OPP"
  # "high_conf-RAND"
)

adaptrates_all=(1)
adst_rmst_pairs_all=(
  "basic-RAND"
)

for seed in "${seeds[@]}"
do
  for alg in "${algs[@]}"
  do
    for pair in "${adst_rmst_pairs_b[@]}"
    do
      IFS='-' read -r adst rmst <<< "$pair"
      for mem_size in "${mem_sizes[@]}"
      do
        for memtype in "${memtypes[@]}"
        do
          for adaptrate in "${adaptrates[@]}"
          do
            CUDA_VISIBLE_DEVICES="$gpu_id" python3 cta_eval.py \
            --seed="$seed" \
            --data="$data" \
            --alg="$alg" \
            --model="$model" \
            --batch_size="$mem_size" \
            --lr="$lr" \
            --device="$device" \
            --workers="$workers" \
            --test_corrupt="$test_corrupt" \
            --eval_mode="$eval_mode" \
            --adaptrate="$adaptrate" \
            --mem_size="$mem_size" \
            --adst="$adst" \
            --rmst="$rmst" \
            --memtype="$memtype" \
            --iabn_k=100000 \
            --iabn_s="$iabn_s" \
            --maxage="$maxage" \
            --confth=0.5 \
            --alginf \
            --iabn \
            --use_tb \
            | tee "$LOG_DIR"/"$seed"_"$alg"_"$adaptrate"_"$adst"_"$rmst"_"$iabn_k"_"$iabn_s"_"$lr"_"$mem_size"_basic.txt &

            # GPU ID update
            gpu_id=$((gpu_id + 1))
            if [ "$gpu_id" -ge "$gpu_count" ]; then
              gpu_id=$initial_gpu_id
            fi
          done
        done
      done
    done
  done
  # wait
done
# wait
for seed in "${seeds[@]}"
do
  for alg in "${algs[@]}"
  do
    for pair in "${adst_rmst_pairs_i[@]}"
    do
      IFS='-' read -r adst rmst <<< "$pair"
      for mem_size in "${mem_sizes[@]}"
      do
        for memtype in "${memtypes[@]}"
        do
          for adaptrate in "${adaptrates[@]}"
          do
            CUDA_VISIBLE_DEVICES="$gpu_id" python3 cta_eval.py \
            --seed="$seed" \
            --data="$data" \
            --alg="$alg" \
            --model="$model" \
            --batch_size="$mem_size" \
            --lr="$lr" \
            --device="$device" \
            --workers="$workers" \
            --test_corrupt="$test_corrupt" \
            --eval_mode="$eval_mode" \
            --adaptrate="$adaptrate" \
            --mem_size="$mem_size" \
            --adst="$adst" \
            --rmst="$rmst" \
            --memtype="$memtype" \
            --iabn_k="$iabn_k" \
            --iabn_s="$iabn_s" \
            --maxage="$maxage" \
            --confth=0.5 \
            --alginf \
            --iabn \
            --use_tb \
            | tee "$LOG_DIR"/"$seed"_"$alg"_"$adaptrate"_"$adst"_"$rmst"_"$iabn_k"_"$iabn_s"_"$lr"_"$mem_size"_iabn.txt &

            # GPU ID update
            gpu_id=$((gpu_id + 1))
            if [ "$gpu_id" -ge "$gpu_count" ]; then
              gpu_id=$initial_gpu_id
            fi
          done
        done
      done
    done
  done
  # wait
done

for seed in "${seeds[@]}"
do
  for alg in "${algs[@]}"
  do
    for pair in "${adst_rmst_pairs_all[@]}"
    do
      IFS='-' read -r adst rmst <<< "$pair"
      for mem_size in "${mem_sizes[@]}"
      do
        for memtype in "${memtypes[@]}"
        do
          for adaptrate in "${adaptrates_all[@]}"
          do
            CUDA_VISIBLE_DEVICES="$gpu_id" python3 cta_eval.py \
            --seed="$seed" \
            --data="$data" \
            --alg="$alg" \
            --model="$model" \
            --batch_size="$mem_size" \
            --lr="$lr" \
            --device="$device" \
            --workers="$workers" \
            --test_corrupt="$test_corrupt" \
            --eval_mode="$eval_mode" \
            --adaptrate="$adaptrate" \
            --mem_size="$mem_size" \
            --adst="$adst" \
            --rmst="$rmst" \
            --memtype="$memtype" \
            --iabn_k="$iabn_k" \
            --iabn_s="$iabn_s" \
            --maxage="$maxage" \
            --confth=0.5 \
            --alginf \
            --use_tb \
            | tee "$LOG_DIR"/"$seed"_"$alg"_"$adaptrate"_"$adst"_"$rmst"_"$iabn_k"_"$iabn_s"_"$lr"_"$mem_size"_basic.txt &

            # GPU ID update
            gpu_id=$((gpu_id + 1))
            if [ "$gpu_id" -ge "$gpu_count" ]; then
              gpu_id=$initial_gpu_id
            fi
          done
        done
      done
    done
  done
done

wait