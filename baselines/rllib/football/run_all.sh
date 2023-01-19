#!/bin/bash

for i in {1..8} 
do
    echo running python3 football.py --expr_name=football-$i --num_workers=$((i*200)) --num_gpus=$i --num_iters 20 
    python3 football.py --expr_name=football-$i --num_workers=$((i*200)) --num_gpus=$i --num_iters 20 
    sleep 1
done