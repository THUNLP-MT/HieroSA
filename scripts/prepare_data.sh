#!/bin/bash

python -m src.prepare_data \
    --path_image /path/to/images \
    --path_train_data train.parquet \
    --path_val_data val.parquet \
