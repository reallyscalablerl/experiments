#!/bin/bash

for i in {1..8}
do
    echo running python3 gym_mujoco.py --expr_name=mujoco-$i --num_workers=$((i*250)) --num_gpus=$i --num_iters 20
    python3 gym_mujoco.py --expr_name=mujoco-$i --num_workers=$((i*250)) --num_gpus=$i --num_iters 20
    sleep 1
done