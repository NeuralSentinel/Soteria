#!/bin/bash

get_available_gpu() {
    gpu_id=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{if(min==""){min=$1; min_idx=NR-1} else if($1<min){min=$1; min_idx=NR-1}} END{print min_idx}')
    echo $gpu_id
}

# dataset=('capitalize_first_letter' 'capitalize_last_letter' 'capitalize_second_letter')
# dataset=('ag_news' 'antonym' 'country-capital' 'country-currency' 'sentiment')
dataset=('ag_news')
# dataset=('antonym' 'country-capital' 'country-currency' 'sentiment')
# dataset=('sentiment') 
# language=('English' 'Hindi' 'Thai' 'German' 'Spanish' 'French' 'Chinese' 'Tamil')
# language=('Arabic' 'Bengali' 'Telugu' 'Bulgarian')
language=('Telugu')
# models=('Qwen/Qwen2-7B-Instruct' 'mistralai/Mistral-7B-Instruct-v0.3' 'microsoft/Phi-3.5-mini-instruct')
models=('microsoft/Phi-3.5-mini-instruct')
# path=('Qwen2_7b' 'Mistral_v3' 'Phi')
path=('Phi')

for i in "${!models[@]}"; do
    for data in "${dataset[@]}"; do
        for lang in "${language[@]}"; do
            # gpu=$(get_available_gpu)
            echo Model_${models[i]}
            echo Dataset_${data}_${lang}
            echo Result_File_Location_../results/${path[i]}_${data}/
            # echo "Running ${data}_${lang} on GPU ${gpu}"
            CUDA_VISIBLE_DEVICES=1 python compute_indirect_effect.py --dataset_name ${data}_${lang} --model_name ${models[i]} --last_token_only False --save_path_root ../results/${path[i]}_${data}/
            # sleep 2
            # echo Sayan_${data}_____${lang}_hjkhksd
        done
    done
done

# CUDA_VISIBLE_DEVICES=0 python compute_indirect_effect.py --dataset_name antonym_French --model_name meta-llama/Llama-3.1-8B-Instruct --last_token_only False --save_path_root ../results/Llama3_1_antonym/

# CUDA_VISIBLE_DEVICES=0 python compute_indirect_effect.py --dataset_name ag_news_English --model_name Qwen/Qwen2-7B-Instruct --last_token_only False --save_path_root ../results/Qwen2_7b_ag_news/