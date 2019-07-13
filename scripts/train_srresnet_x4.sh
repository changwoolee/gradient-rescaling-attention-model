#!/bin/bash
#python3 train.py -p=sr-resnet -s=4 $@
python3 train.py -p=sr-resnet -s=4 --pred_logvar 
python3 train.py -p=sr-resnet -s=4 --pred_logvar --attention
python3 train.py -p=sr-resnet -s=4 --pred_logvar --attention --block_attention_gradient

