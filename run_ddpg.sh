#!/bin/bash
for i in 0 1 2 3 4; do
  python train_ddpg.py --cuda -n "DDPG_VSS-v0_0$i"
done
