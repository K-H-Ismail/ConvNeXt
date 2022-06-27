NUM_PROC=2
echo 'using' $NUM_PROC 'gpu(s)'
BATCHSIZE=256
UP_FREQ=$((4096 / (BATCHSIZE * NUM_PROC)))

echo 'using' $UP_FREQ 'updates'

time=$(date '+%d_%m_%Y_%H:%M:%S')
dataset='ImageNet192'
root='/home/ismail'
model=convnext_tiny

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port 47781 $root/ConvNeXt/main.py \
--model $model --drop_path 0.1 --batch_size $BATCHSIZE --lr 4e-3 \
--update_freq $UP_FREQ --input_size 224 --nb_classes 1000 \
--data_path $root/datasets/$dataset/2012 \
--output $root/ConvNeXt/outputs/output_${model}_${dataset}_${time} \
--use_dcls 'yes' --dcls_kernel_size 17 --dcls_kernel_count 34 --dcls_sync 'yes' --use_loss_rep 'yes' \
--seed 0 --enable_wandb 'yes' \
#--auto_resume 'yes' --output_dir '/gpfswork/rech/owj/uvr14im/ConvNeXt/outputs/output_convnext_tiny_ImageNet_03_06_2022_15:04:03' \
#--model_ema true --model_ema_eval true \

