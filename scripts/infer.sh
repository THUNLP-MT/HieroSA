#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m src.infer \
    --model_path /path/to/checkpoint \
    --path_image /path/to/images \
    --path_output output.json \
    --path_output_image output_images \
    --visualize \
