#!/bin/bash

python3 smac-appo.py --expr_name=smac-1w --num_workers=1 --num_gpus=1 --num_iters 20
sleep 1

for i in {1..4}
do
    echo running python3 smac-appo.py --expr_name=smac-$i --num_workers=$((i*100)) --num_gpus=$i --num_iters 20
    python3 smac-appo.py --expr_name=smac-$i --num_workers=$((i*100)) --num_gpus=$i --num_iters 20
    sleep 1
done