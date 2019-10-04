#!/bin/bash
#python3 train.py -p=sr-resnet -s=4 $@
#python3 train.py -p=sr-resnet -s=4 --pred_logvar -o ./output/logvar
#python3 train.py -p=sr-resnet -s=4 --pred_logvar --attention -o ./output/logvar_attention
python3 train.py -p=sr-resnet -s=4 --pred_logvar --attention --block_attention_gradient -o ./output/logvar_attention_bg_test

