gpu=4
port=$1  # 1234
dataset=pascal
exp_name=split$2  # 0/1/2/3
arch=HMNet
net=$3  # vgg/renet50
postfix=$4  # manet/manet_5s

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}_${postfix}.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train_pascal.sh train_pascal.py ${config} ${exp_dir}

echo ${arch}
echo ${config}

python3 -m torch.distributed.launch --nproc_per_node=${gpu} --master_port=${port} train_pascal.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/train-$now.log
