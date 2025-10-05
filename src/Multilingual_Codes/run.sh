#!/bin/bash

#-------------------------
#Need to change heads folder for different models
#-------------------------

finetune_model=$1
base_model=$2
language=$3
dataset_file=$4
scaling_factor=$5
do_base=$6
batch_size=$7
head_folder=$8
# reso=$9

python Multilingual_Neuron.py -fine ${finetune_model} -base ${base_model} -scale ${scaling_factor} -lang ${language} -head ${head_folder}

python Inference_Safe.py -base ${base_model} -dataset ${dataset_file} -lang ${language} -scale ${scaling_factor} -batch ${batch_size}

if [ "$do_base" = "yes" ]; then
    python Inference_Base.py -base ${base_model} -dataset ${dataset_file} -lang ${language} -scale ${scaling_factor} -batch ${batch_size}
fi

python Multilingual_Eval.py -base ${base_model} -dataset ${dataset_file} -lang ${language} -scale ${scaling_factor} -config Safe

if [ "$do_base" = "yes" ]; then
    python Multilingual_Eval.py -base ${base_model} -dataset ${dataset_file} -lang ${language} -scale ${scaling_factor} -config Base
fi