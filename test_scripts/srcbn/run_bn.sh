#!/bin/bash

# corruption 0 (Gaussian noise) to 14 (JPEG)
date=$(date +'%m%d' | sed 's/^0//;s/0\([0-9]\{2\}\)$/\1/')
initial_cor_id=0
corrupt_count=14


for ((corrupt=$initial_cor_id; corrupt<=$corrupt_count; corrupt++))
do    
    ./run_bn10c.sh $1 $corrupt &
    ./run_bn100c.sh $1 $corrupt &
    ./run_bnIN.sh $1 $corrupt &
    wait
done