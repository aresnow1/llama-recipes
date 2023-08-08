screen -L -Logfile function_call.log \
python llama_finetuning.py --use_peft \
  --peft_method lora \
  --quantization \
  --model_name /new_data1/hekaisheng/.cache/modelscope/hub/modelscope/Llama-2-13b-chat-ms/ \
  --output_dir ./output_model \
  --dataset function_call_dataset \
  --batch_size_training 16 \
  --micro_batch_size 4 \
  --num_epochs 1
