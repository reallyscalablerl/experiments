#!/bin/bash

for i in {1..8}
do
    echo running python3 atari-appo.py --expr_name=atari-$i --num_workers=$((i*32)) --num_gpus=$i --num_iters 20
    python3 atari-appo.py --expr_name=atari-$i --num_workers=$((i*32)) --num_gpus=$i --num_iters 20
    sleep 1
done