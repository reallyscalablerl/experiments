#!/bin/bash

SURREAL_CONFIG_PATH=.surreal.yml python3 -m surreal.subproc.surreal_subproc experiment_name --algorithm ppo --num_agents 128 --env gym:Humanoid-v3