#!/bin/bash

set -e

CHECKPOINT="pretrained_models/checkerboard-L-2x/checkpoint-last.pth"
CFG=1.5
CFG_START_STEP=5
NUM_STEPS_PER_SCALE=4

NUM_SAMPLES=50000

k="cfg${CFG}_${CFG_START_STEP}_steps${NUM_STEPS_PER_SCALE}"

base_dir=$(dirname "$0")
samples_dir="$base_dir/eval/${k}_samples"
result_file="$base_dir/eval/${k}_eval.txt"

mkdir -p $samples_dir

python main_evaluate.py \
        --checkpoint $CHECKPOINT \
        --cfg $CFG \
        --cfg_start_step $CFG_START_STEP \
        --steps_per_scale $NUM_STEPS_PER_SCALE \
        --keep_samples \
        --samples_only \
        --eval_num_samples $NUM_SAMPLES \
        --samples_output_dir $samples_dir

python evaluator.py VIRTUAL_imagenet256_labeled.npz $samples_dir/*/256 --output_file $result_file

rm -r $samples_dir
