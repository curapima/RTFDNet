#!/bin/bash

CONFIGS=(
    "configs/segformer/EAEF_mit-b2_1xb8-40K_MF_512x512.py"
    # "configs/segformer/EAEF_mit-b2_1xb8-40K_FMB_512x512.py"
    # "configs/segformer/EAEF_mit-b2_1xb8-40K_PST_512x512.py"

)

WORKDIR_BASE="./work_dirs"

for cfg in "${CONFIGS[@]}"; do
    echo "==============================="
    echo "Training with config: $cfg"
    echo "==============================="

    python tools/train.py \
        --config "$cfg" \
        --work-dir "$WORKDIR_BASE/$(basename "$cfg" .py)" \
        #--amp --resume
done

#echo "All configs finished. Shutting down..."
#/usr/bin/shutdown now
