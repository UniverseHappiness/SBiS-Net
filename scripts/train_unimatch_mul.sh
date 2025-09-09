#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='acdc'
method='unimatch_reconstruction_channel_exchange'
exp='unet'
config=configs/$dataset.yaml

# 自由定义 split 列表（可以是 '1' '3' '7' 或任意组合）
split_list=('3' '7')  # 可以改成 ('1' '3' '7') 或 ('3') 等任意组合

for split in "${split_list[@]}"; do
    labeled_id_path=splits/$dataset/$split/labeled.txt
    unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
    save_path=exp/$dataset/$method/$exp/$split

    mkdir -p $save_path

    echo "Running split=$split, saving to $save_path"
    python -m torch.distributed.launch \
        --nproc_per_node=$1 \
        --master_addr=localhost \
        --master_port=$2 \
        $method.py \
        --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
        --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
done

# 说明：
# dataset: ['acdc']
# method: ['unimatch', 'supervised']
# exp: 仅用于指定 'save_path'
# split_list: 可自由调整，如 ('1' '3' '7') 或 ('3') 等