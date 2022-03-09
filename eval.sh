python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--model convnext_large --eval true \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth \
--input_size 224 --drop_path 0.2 \
--data_path /home/ismail/ImageNet/2012 \
--nb_classes 1000
