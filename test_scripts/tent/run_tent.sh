#!/bin/bash

# corruption 0 (Gaussian noise) to 14 (JPEG)
date=$(date +'%m%d' | sed 's/^0//;s/0\([0-9]\{2\}\)$/\1/')
initial_cor_id=0
corrupt_count=14

for ((corrupt=$initial_cor_id; corrupt<=$corrupt_count; corrupt++))
do
    ./run_tent10c_0_bs.sh $1 $corrupt &
    ./run_tent100c_0_bs.sh $1 $corrupt &
    ./run_tentIN_0_16.sh $1 $corrupt &
    wait
done