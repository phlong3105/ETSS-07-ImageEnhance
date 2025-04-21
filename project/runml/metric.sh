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

# Input
task="llie"
arch=""
model=""
data=(

)
device="cuda:0"

# Run
cd "${runml_dir}" || exit
for (( i=0; i<${#data[@]}; i++ )); do
    input_dir="${data_dir}/${task}/#predict/${arch}/${model}/${data[i]}"
    if ! [ -d "${input_dir}" ]; then
        input_dir="${current_dir}/run/predict/${arch}/${model}/${data[i]}"
    fi
    target_dir="${data_dir}/enhance/${task}/${data[i]}/test/ref"
    if ! [ -d "${target_dir}" ]; then
        target_dir="${data_dir}/enhance/${task}/${data[i]}/val/ref"
    fi

    if [ -d "${target_dir}" ]; then
        python -W ignore metric.py \
          --input-dir "${input_dir}" \
          --target-dir "${target_dir}" \
          --result-file "${current_dir}" \
          --arch "${arch}" \
          --model "${model}" \
          --data "${data[i]}" \
          --device "${device}" \
          --imgsz 512 \
          --metric "psnr" \
          --metric "ssimc" \
          --metric "psnry" \
          --metric "ssim" \
          --metric "ms_ssim" \
          --metric "lpips" \
          --metric "brisque" \
          --metric "ilniqe" \
          --metric "niqe" \
          --metric "pi" \
          --backend "pyiqa" \
          --use-gt-mean
    else
        python -W ignore metric.py \
          --input-dir "${input_dir}" \
          --target-dir "${target_dir}" \
          --result-file "${current_dir}" \
          --arch "${arch}" \
          --model "${model}" \
          --data "${data[i]}" \
          --device "${device}" \
          --imgsz 512 \
          --metric "psnr" \
          --metric "ssimc" \
          --metric "psnry" \
          --metric "ssim" \
          --metric "ms_ssim" \
          --metric "lpips" \
          --metric "brisque" \
          --metric "ilniqe" \
          --metric "niqe" \
          --metric "pi" \
          --backend "pyiqa"
    fi

done

# Done
cd "${current_dir}" || exit
