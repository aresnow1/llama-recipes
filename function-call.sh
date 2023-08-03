screen -L -Logfile function_call.log \
python llama_finetuning.py --use_peft \
  --peft_method lora \
  --quantization \
  --model_name meta-llama/Llama-2-7b-hf \
  --output_dir ./output_model \
  --dataset financial_dataset \
  --batch_size_training 16 \
  --micro_batch_size 4 \
  --num_epochs 1
