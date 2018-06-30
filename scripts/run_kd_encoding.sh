#!/bin/bash
#############################################################################
# This script is used for the KD encoding methods.
# Options:
#   model: size of model.
#   run_mode: random code, pretrained code, end2e wo/w guidances.
# Any other options should be leaved as default.
#############################################################################
cd ..
gpu=$1
echo $gpu

model=small
#model=medium
#model=large

#run_mode=random_code
#run_mode=pretr_code
#run_mode=no_guide
#run_mode=online_guide
run_mode=emb_guide

#for run_mode in random_code pretr_code no_guide online_guide emb_guide; do
dataset=ptb
data_path=data
save_root=results
rnn_mode=block
use_recoding=True
code_type=redundant
rnn_residual=False
ec_hard_code_output=True
ec_STE_softmax_transform=True
ec_logits_bn=1.0
ec_code_dropout=0.
ec_emb_baseline_dropout=0.
ec_emb_autoencoding=False
ec_emb_transpose_layers=1
ec_emb_transpose_dim=200
ec_emb_transpose_actv=relu

if [ $model == small ]; then
	max_max_epoch=15
	max_grad_norm=3
else
	max_max_epoch=35
	max_grad_norm=5
fi

for KD in 32,32; do
IFS=","; set -- $KD; K=$1; D=$2;
for compound in none,1000,1,1e-2; do
IFS=","; set -- $compound; method=$1; step=$2; rate=$3; bound=$4
for emb_dim in 300 ; do
#for emb_dim in 100 200 300; do
for ec_code_generator in STE_argmax ; do
#for ec_code_generator in preassign STE_argmax gumbel_softmax; do
for learning_rate in 1 ; do
for optimizer in mixed ; do
#for optimizer in lazy_adam sgd scheduled_sgd momentum; do
for ec_aggregator in mean ; do
for ec_emb_baseline in False ; do
for ec_emb_baseline_reg in 0; do
for ec_entropy_reg in 0; do
	ec_code_emb_dim=$emb_dim
	ec_fnn_hidden_size=300
	ec_fnn_hidden_actv=linear
	ec_rnn_num_layers=1
	ec_rnn_bidirection=False
	#ec_rnn_hidden_size=300
	ec_rnn_hidden_size=500
	ec_shared_coding=False
	ec_temperature_decay_method=$method
	ec_temperature_decay_steps=$step
	ec_temperature_decay_rate=$rate
	ec_temperature_lower_bound=$bound
	ec_emb_baseline_reg2=0
	ec_emb_baseline_reg3=0

	code_load_filename=
	emb_load_filename=
	if [ $run_mode == random_code ]; then
		ec_code_generator=preassign
		optimizer=scheduled_sgd
	elif [ $run_mode == pretr_code ]; then
		code_load_filename=$save_root/pretrains/$model\_embs.code.K$K\D$D.pkl
		ec_code_generator=preassign
		optimizer=scheduled_sgd
	elif [ $run_mode == online_guide ]; then
		ec_emb_baseline=True
		if [ $model == small ]; then
			ec_emb_baseline_reg=0.1
			ec_emb_baseline_dropout=0.5
		else
			ec_emb_baseline_reg=1000
			ec_emb_baseline_dropout=0.3
		fi
	elif [ $run_mode == emb_guide ]; then
		ec_emb_baseline=True
		ec_emb_baseline_reg=1000
		if [ $ec_emb_autoencoding == True ] || [ $ec_emb_autoencoding == true ]; then
			ec_emb_baseline_reg2=100
			ec_emb_baseline_reg3=10
		fi
		emb_load_filename=$save_root/pretrains/$model\_embs.pkl
	fi

	save_path=$save_root/kd_encoding/$(date +"%m%d_%H%M%S")\_m$model\_r$run_mode\_K$K\_D$D\_cg$ec_code_generator\_cshare$ec_shared_coding\_hard$ec_hard_code_output\_logitsbn$ec_logits_bn\_aggregator$ec_aggregator\_optimizer$optimizer\_dim$emb_dim\_decay$method\_decaystep$step\_decayrate$rate\_lr$learning_rate\_maxgradn$max_grad_norm\_basereg$ec_emb_baseline_reg\_entreg$ec_entropy_reg
	echo $save_path
	if [ -e $save_path ]; then
		rm -r $save_path
	fi
	mkdir -p $save_path
	CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python ptb_word_lm.py --dataset=$dataset --data_path=$data_path --model=$model --use_recoding=$use_recoding --rnn_mode=$rnn_mode --optimizer=$optimizer --save_path=$save_path --max_max_epoch=$max_max_epoch --max_grad_norm=$max_grad_norm --K=$K --D=$D --ec_logits_bn=$ec_logits_bn --ec_rnn_num_layers=$ec_rnn_num_layers --ec_rnn_bidirection=$ec_rnn_bidirection --ec_code_generator=$ec_code_generator --code_type=$code_type --ec_STE_softmax_transform=$ec_STE_softmax_transform --ec_hard_code_output=$ec_hard_code_output --ec_aggregator=$ec_aggregator --ec_code_emb_dim=$ec_code_emb_dim --ec_rnn_hidden_size=$ec_rnn_hidden_size --ec_shared_coding=$ec_shared_coding --ec_temperature_decay_method=$ec_temperature_decay_method --ec_temperature_decay_steps=$ec_temperature_decay_steps --ec_temperature_decay_rate=$ec_temperature_decay_rate --code_load_filename=$code_load_filename --rnn_residual=$rnn_residual --learning_rate=$learning_rate --ec_code_dropout=$ec_code_dropout --ec_emb_baseline=$ec_emb_baseline --ec_emb_baseline_reg=$ec_emb_baseline_reg --ec_emb_baseline_reg2=$ec_emb_baseline_reg2 --ec_emb_baseline_reg3=$ec_emb_baseline_reg3 --emb_load_filename=$emb_load_filename --ec_entropy_reg=$ec_entropy_reg --ec_emb_autoencoding=$ec_emb_autoencoding --ec_temperature_lower_bound=$ec_temperature_lower_bound --ec_emb_baseline_dropout=$ec_emb_baseline_dropout >$save_path/log 2>&1 &
	#let "gpu+=1"
done
done
done
done
done
done
done
done
done
done
#done
