#!/bin/bash

for dirname in "smac" # "atari" "football" # "mujoco" 
do
    cd $dirname
    ./run_all.sh
    cd ..
done
