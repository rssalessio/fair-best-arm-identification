#!/bin/bash

# Loop directly over the delta values



for DELTA in 0.001 0.01 0.1; do
    python synthetic_mp_simulations_prespecified.py "$DELTA"
    python synthetic_mp_simulations_thetadep.py "$DELTA"
    python scheduling_mp_simulations_prespecified.py "$DELTA"
    python scheduling_mp_simulations_thetadep.py "$DELTA"
done
