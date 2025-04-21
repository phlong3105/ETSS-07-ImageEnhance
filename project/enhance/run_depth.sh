#!/bin/bash
echo "$HOSTNAME"
clear

# Directories
current_file=$(readlink -f "$0")
current_dir=$(dirname "$current_file")
project_dir=$(dirname "$current_dir")
mon_dir=$(dirname "$project_dir")
runml_dir="${project_dir}/runml"
data_dir="${mon_dir}/data"

# Run
cd "${runml_dir}" || exit
python -W ignore main.py \
    --root "${current_dir}" \
    --task "depth" \
    --mode "predict" \
    --arch "depth_anything_v2" \
    --model 4 \
    --config 0 \
    --weights 0 \
    --data "" \
    --verbose \
    "$@"

# Done
cd "${current_dir}" || exit
