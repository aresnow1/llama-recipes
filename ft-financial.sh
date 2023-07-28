#!/bin/bash

python llama_finetuning.py  --use_peft --peft_method lora --quantization --model_name meta-llama/Llama-2-7b-hf --output_dir ./output_model --dataset financial_dataset