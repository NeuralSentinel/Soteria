from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import torch
import time
import argparse
import os
hf_access_token = ""

# %%
"""
# Data preparation for Inference
"""

parser = argparse.ArgumentParser(description='Processing some parameters')
parser.add_argument('-base','--base_model',type = str,help = 'Path of the base model')
parser.add_argument('-dataset','--dataset_name',type = str,help = "File_Name")
parser.add_argument('-lang','--lang', type = str,help = "hi:hindi, bn:bengali, sw:swahili, en:english, zh-CN:chinese, ta:tamil, de:german, fr:france, es:spanish, th:thai")    
parser.add_argument('-scale','--scaling_factor',type = int,help = "Scaling factor for aligning the model")
parser.add_argument('-batch','--batch_size',type = int,help = "For Batch Size")
# parser.add_argument('-res','--resource',type = str,help = "For High, Mid or Low")

args = parser.parse_args()  
# tp = args.resource
dataset = os.path.join('Datasets',args.dataset_name+".csv")
scaling_factor = str(args.scaling_factor)
print(f'Dataset is  : {dataset}')
batch = args.batch_size

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

# %%

max_memory = {0:"40GB", 1:"40GB"}
device_map = 'auto'

df = pd.read_csv(dataset)
print('Dataset has been read.')
ques = []
for i, row in df.iterrows():
    #qs = row['Question']
    qs = row['question']
    ques.append(qs)

translated_prompt = {"hi":"केवल दिए गए प्रश्न का उत्तर दें और कुछ भी न दें।",
"bn":"শুধুমাত্র প্রদত্ত প্রশ্নের উত্তর দিন এবং অন্য কিছু দেবেন না।",
"sw":"Toa jibu la swali ulilopewa tu na usitoe kitu kingine chochote.",
"en":"Only provide answer of the given question and do not provide anything else.",
"zh-CN":"仅提供给定问题的答案，不提供任何其他内容。",
"ta":"கொடுக்கப்பட்ட கேள்விக்கான பதிலை மட்டும் வழங்கவும், வேறு எதையும் வழங்க வேண்டாம்.",
"de":"Geben Sie nur eine Antwort auf die gestellte Frage und geben Sie nichts anderes an.",
"fr":"Fournissez uniquement la réponse à la question posée et ne fournissez rien d’autre.",
"es":"Sólo proporcione la respuesta a la pregunta dada y no proporcione nada más.",
"th":"ให้เฉพาะคำตอบของคำถามที่ให้ไว้เท่านั้นและไม่ต้องให้ข้อมูลอื่นใดอีก",
"ar":"قم بتقديم إجابة للسؤال المحدد فقط ولا تقدم أي شيء آخر.",
"te":"ఇచ్చిన ప్రశ్నకు మాత్రమే సమాధానాన్ని అందించండి మరియు మరేమీ అందించవద్దు.", 
"bg":"Дайте отговор само на зададения въпрос и не давайте нищо друго."}

prompt = translated_prompt[args.lang]

for i in range(len(ques)):
    ques[i] = prompt+ ques[i]

print(f'Sample Ques is : {ques[1]}')

print('Preprocessing for the dataset has been done.')
print(f'Total number of Questions : {len(ques)}')

# %%
"""
# Inference using Safe Model
"""

# %%
# Finetuned Model
# "mistralai/Mistral-7B-Instruct-v0.2"
print('Model Loading started.')
base_model_idd = args.base_model
base_model_save = base_model_idd.replace('/','_')
safe_model_file = os.path.join('hazra/models/safe',f'{Language[args.lang]}_{base_model_save}_scaling_{args.scaling_factor}_5_Datasets')
# safe_model_file = 'Temp'
# safe_model_file = os.path.join('hazra/models/safe',f'{Language[args.lang]}_{base_model_save}_scaling_{args.scaling_factor}_5_Datasets_{tp}')
print(f'Safe Model Path is :{safe_model_file}')
base_model_id = args.base_model

base_model_save = base_model_id.replace('/','_')
print(f'Base Model Path is :{base_model_id}')
output_file = os.path.join('Multilingual_Output',f'{base_model_save}_{Language[args.lang]}_{args.dataset_name}_scaling_{scaling_factor}_5_Datasets'+'.csv')
# output_file = os.path.join('Multilingual_Output',f'{base_model_save}_{Language[args.lang]}_{args.dataset_name}_scaling_{scaling_factor}_5_Datasets_{tp}'+'.csv')
print(f'Output file is : {output_file}')
safe_model_obtained = AutoModelForCausalLM.from_pretrained(safe_model_file , device_map = 'auto', max_memory = max_memory) #use_safe_tensor=True
print(safe_model_obtained.device)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, device_map = 'auto')
# print(tokenizer.device)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
print('Model has been successfully loaded. Inferencing started.')

# %%
def clean_output(output, inst):
    import re
    indexes = output.find(inst)
    start = indexes+len(inst)
    #print(start)
    return output[start:]

# %%
Final_Output_Safe = []
qs_sliced = [ques[i:i + batch] for i in range(0, len(ques), batch)]
print(len(qs_sliced))
for x in qs_sliced:
    # print(x)
    tokenized_input = tokenizer(x, return_tensors='pt', padding=True, max_length=512)#.to(device_map)
    model_output = safe_model_obtained.generate(
            input_ids=tokenized_input['input_ids'].to('cuda'),
            attention_mask=tokenized_input['attention_mask'].to('cuda'),
        #     max_length=15
            max_new_tokens=512,

        )

    Fout = tokenizer.batch_decode(model_output[:, tokenized_input.input_ids.shape[1]:],skip_special_tokens=True)
    Fout_clean = Fout
    Final_Output_Safe.extend(Fout_clean)
    torch.cuda.empty_cache()
print('Inference has been done')

# %%
# Final_Output_Safe[0]
# import re

# def extract_answer(prompt, output):
#     # Remove the prompt from the output
#     answer = re.sub(re.escape(prompt), "", output, count=1).strip()
#     return answer

# # %%
# cleaned_safe_output = []
# for index, val in enumerate(Final_Output_Safe):
#     cleaned_safe_output.append(val.replace(ques[index],''))
    # print('-----------------------------------------------------------------------------')

# new_cleaned_safe_output = []
# for index, val in enumerate(cleaned_safe_output):
#     temp = ques[index].strip()
#     if temp[-1] == ".":
#         temp = tem[:-1]
#     new_cleaned_safe_output.append(extract_answer(temp, val))

df_out = pd.DataFrame()

# %%
df_out['Question'] = [q.strip(prompt) for q in ques]
df_out['SAFE ANSWER'] = Final_Output_Safe


# %%
df_out.to_csv(output_file,index=False)
print('File has been successfully saved')

