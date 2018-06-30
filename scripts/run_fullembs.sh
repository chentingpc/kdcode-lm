#!/bin/bash
#########################################################################
# This script is used to generate full embedding and low-rank baselines.
# Options:
#   save_embedding: to save the pre-trained embedding when using
#     full embedding baseline, set it to true.
#   emb_lowrank_dim: to enable low-rank baseline, set emb_lowrank_dim
#     to any float number between 0 and 1, denoting keep params rate.
#########################################################################
cd ..
gpu=$1
echo GPU=$gpu

save_model_secs=600
use_recoding=False
dataset=ptb
data_path=data/
save_root=results/
rnn_mode=block
optimizer=sgd
#save_embedding=true
save_embedding=false

for model in small ; do
#for model in small medium large; do
for emb_lowrank_dim in 0; do
	if [ $save_embedding == true ]; then
		emb_save_filename=$save_root/$dataset\_$model\_embs.pkl
		echo "Will save embedding to $emb_save_filename"
	else
		echo "Won't save embedding"
	fi
	if [ $model == small ]; then
		max_max_epoch=15
		max_grad_norm=5
	elif [ $model == medium ]; then
		max_max_epoch=30
		max_grad_norm=5
	else
		max_max_epoch=35
		if [ $emb_lowrank_dim != 0 ]; then
			max_grad_norm=5
		else
			max_grad_norm=10
		fi
	fi
	save_path=$save_root/baselines/model$model\_rnnmode$rnn_mode\_optimizer$optimizer\_maxgradn$max_grad_norm\_lowrankd$emb_lowrank_dim
	echo $save_path
	if [ -e $save_path ]; then
		rm -r $save_path
	fi
	mkdir -p $save_path
	CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python ptb_word_lm.py --dataset=$dataset --data_path=$data_path --model=$model --use_recoding=$use_recoding --rnn_mode=$rnn_mode --optimizer=$optimizer --save_path=$save_path --max_max_epoch=$max_max_epoch --max_grad_norm=$max_grad_norm --emb_save_filename=$emb_save_filename --emb_lowrank_dim=$emb_lowrank_dim --save_model_secs=$save_model_secs >$save_path/log 2>&1 #&
done
done
