# Soteria: Language-Specific Functional Parameter Steering for Multilingual Safety Alignment

### 🎉 Accepted at EMNLP-2025 (Long Paper)

Soteria is a method designed for advancing **multilingual safety alignment** via **language-specific functional parameter steering**.  
Our work aims to enhance the safety and robustness of multilingual language models by providing high-quality aligned data across languages.

---

## 📂 Dataset
We release our dataset on Hugging Face:  👉 [SoftMINER-Group/Soteria](https://huggingface.co/datasets/SoftMINER-Group/Soteria)

---

## 📄 Paper
Read our paper here:  👉 [Read the paper](https://arxiv.org/abs/2502.11244)

---

## 🚀 How to Run the Code

Follow these steps to run the main components of this repository:

**Step 1:**
Go to the following folder location:
```
/src
```
Run the following command to get the activation heads:
```
python compute_indirect_effect.py --dataset_name {Dataset_Name} --model_name {Model_Name} --last_token_only False
```

**Step 2:**
Run the following notebook to get the top-k activation heads:
```
src/chek_indirect_ST_get_top_heads.ipynb
```

**Step 3:**
For getting the safe model, performing inference, and evaluation, run the following script:
```
src/Multilingual_Codes/run.sh
```
Command format:
```
bash ./run.sh {finetune_model} {base_model} {language} {dataset_file} {scaling_factor} {do_base} {batch_size} {head_folder}
```

Replace the placeholders in curly braces with your specific arguments.

---

## 📜 Citation
If you find this resource useful, please cite our work:

```bibtex
@misc{banerjee2025soterialanguagespecificfunctionalparameter,
      title={Soteria: Language-Specific Functional Parameter Steering for Multilingual Safety Alignment}, 
      author={Somnath Banerjee and Sayan Layek and Pratyush Chatterjee and Animesh Mukherjee and Rima Hazra},
      year={2025},
      eprint={2502.11244},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11244}, 
}
```
