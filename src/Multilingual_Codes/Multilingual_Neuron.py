from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import torch
import time
import argparse
import os
from task_vectors import TaskVector
hf_access_token = ""

# %%
"""
# Data preparation for Inference
"""

parser = argparse.ArgumentParser(description='Processing some parameters')
parser.add_argument('-fine','--finetune_model',type = str, help = 'Path of the fine tune model')
parser.add_argument('-base','--base_model',type = str,help = 'Path of the base model')
parser.add_argument('-scale','--scaling_factor',type = int,help = "Scaling factor for aligning the model")
parser.add_argument('-lang','--lang', type = str,help = "hi:hindi, bn:bengali, sw:swahili, en:english, zh-CN:chinese, ta:tamil, de:german, fr:france, es:spanish, th:thai") 
parser.add_argument('-head','--head', type = str, help = "Folder containing the head of the given model")
# parser.add_argument('-res','--resource',type = str,help = "For High, Mid or Low")
args = parser.parse_args()  
# tp = args.resource
# %%
device_map = "auto"
max_memory ={0:"80GB"}

Language = {"hi":"Hindi",
"bn":"Bengali",
"sw":"Swahili",
"en":"English",
"zh-CN":"Chinese",
"ta":"Tamil",
"de":"German",
"fr":"French",
"es":"Spanish",
"th":"Thai",
"ar":"Arabic",
"bn":"Bengali", 
"te":"Telugu", 
"bg":"Bulgarian"}
base_model_idd = args.base_model
base_model_save = base_model_idd.replace('/','_')
safe_model_file = os.path.join('hazra/models/safe',f'{Language[args.lang]}_{base_model_save}_scaling_{args.scaling_factor}_5_Datasets')
# safe_model_file = 'Temp'
# safe_model_file = os.path.join('hazra/models/safe',f'{base_model_save}_scaling_{args.scaling_factor}_5_Datasets_{tp}')
head = args.head
head_file = os.path.join('Heads',f'{head}_5_Dataset',f'{Language[args.lang]}_Heads.json')
# head_file = os.path.join('Heads',f'{head}_5_Dataset',f'{Language[args.lang]}_100_Heads.json')
# head_file = os.path.join('Heads',f'{head}_5_Dataset',f'{tp}_Lang.json')

print(f'Safe_Model_File : {safe_model_file}')
print(f'Head_file : {head_file}')

if os.path.exists(safe_model_file) and os.path.isdir(safe_model_file):
    print("Folder exists.")
    exit(0)
else:
    print("Folder does not exist.")

# %%
"""
## Model preprocessing
"""

finetuned_model_id = args.finetune_model
print(f'Fine Tuned Model is : {finetuned_model_id}')
finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_id, use_safetensors = True)#, max_memory = max_memory, device_map = 'auto')

# %%
# Pretrained Model
base_model_id = args.base_model
print(f'Base model is : {base_model_id}')
base_model = AutoModelForCausalLM.from_pretrained(base_model_id,  use_safetensors = True)#, max_memory = max_memory, device_map = 'auto') #, device_map = "auto")

harm_task_vector = TaskVector(base_model, finetuned_model)
# harm_task_vector = base_model.state_dict()
del finetuned_model
# %%
import json
    
with open(head_file, 'r') as file:
    data = json.load(file)
# top_heads = data[0:500]
top_heads = data

len(top_heads)

harm_vector = harm_task_vector.get_print()

mask = {}
for key, item in harm_vector.items():
    mask[key] = np.zeros(item.shape)

# %%
for item in top_heads:
    layer_num= item[0]
    # layer_num = int(item.replace('\n','').split(',')[0].replace('(',''))
    head_num = item[1]
    # head_num = int(item.replace('\n','').split(',')[1])
    # head_num = int(item.replace('\n','').split(',')[1].replace(')','').lstrip())
    cur_item = mask[f'model.layers.{layer_num}.self_attn.o_proj.weight']
    cur_item[:,(head_num*128):(head_num+1)*128] = 1
    mask[f'model.layers.{layer_num}.self_attn.o_proj.weight'] = cur_item
    # print(layer_num, head_num)
    # print(head_num)
    # for i, it in enumerate(cur_item):
    #     print(i)
    #     print(it)
    # break

# %%
# # ## For checking : - Not Done

# for item in top_heads:
#     # layer_num= int(item.replace('\n','').split(',')[0].replace('(',''))
#     layer_num = item[0]
#     # head_num = int(item.replace('\n','').split(',')[1])
#     # head_num = int(item.replace('\n','').split(',')[1].replace(')','').lstrip())
#     head_num = item[1]
#     print(layer_num)
#     print(head_num)
#     for i, x in enumerate(mask[f'model.layers.{layer_num}.self_attn.o_proj.weight']):
#         print(i)
#         print(x[head_num*128-1:(head_num+1)*128+1])
        
#     break

# %%
print('Performing operation for masking : -')
harm_task_vector.calc_Special_Vector(mask, harm_vector)

print('Obtained the final masked difference matrix')

edit_layer_harm_vector = harm_task_vector.get_special_vec()

print('Performing difference matrix')
safe_model = harm_task_vector.apply_special_matrix(base_model, edit_layer_harm_vector, scaling_coef=float(args.scaling_factor))
print('Obtained final difference matrix')

safe_model.state_dict()

safe_model.save_pretrained(safe_model_file)
print('Model has been saved successfully')

