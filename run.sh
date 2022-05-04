time=$(date '+%d_%m_%Y_%H:%M:%S')
dataset='ImageNet48'
root='/home/ismail'
model=convnext_tiny

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 47781 ./main.py \
--model $model --drop_path 0.1 --batch_size 256 --lr 4e-3 \
--update_freq 8 --input_size 224 --nb_classes 1000 \
--data_path $root/datasets/$dataset/2012 \
--output $root/ConvNeXt/outputs/output_${model}_${dataset}_${time} \
--seed 0 --enable_wandb 'yes' \
--use_dcls 'yes' --dcls_kernel_size 7 --dcls_kernel_count 7 --dcls_sync 'yes' --use_loss_rep 'yes'
#--auto_resume true --resume "/home/ismail/ConvNeXt/outputs/output_convnext_tiny_ImageNet48_28_02_2022_15:31:04/checkpoint-42.pth" \
#--model_ema true --model_ema_eval true \
