NUM_PROC=2
echo 'using' $NUM_PROC 'gpu(s)'
dataset='ImageNet'
root='/home/ismail'
model=convnext_tiny
output=output_convnext_tiny_ImageNet_11_06_2022_22:37:36

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port 47781 $root/ConvNeXt/main.py \
--model $model --eval true --drop_path 0.1 --input_size 224 --nb_classes 1000 \
--data_path $root/datasets/$dataset/2012 \
--resume $root/ConvNeXt/outputs/$output/checkpoint-best-ema.pth \
--use_dcls 'yes' --dcls_kernel_size 17 --dcls_kernel_count 34 --dcls_sync 'yes' --use_loss_rep 'yes' \
--model_ema true --model_ema_eval true \





