#!/bin/bash
# waits for the A100 campaign, then runs the device microbenchmarks
while ! grep -q CAMPAIGN_DONE <WS>/logs/gpu_campaign_a100.log; do sleep 60; done
RR_SCRIPT=gpu_micro.py RR_PIP="cupy-cuda12x cuquantum-python-cu12" \
  <WS>/bench/gpu/run_remote.sh a100_micro gpu-node gpu:a100:1 02:00:00 --
