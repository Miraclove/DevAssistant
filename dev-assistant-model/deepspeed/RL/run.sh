CUDA_VISIBLE_DEVICES=2 python sft.py \
    --model_name '/home/uework/AiWeb/pwz/github/model/starcoderbase-7b'\
    --dataset_name '/home/uework/AiWeb/pwz/github/data/dataset/shengqu-lua2-train'\
    --learning_rate 1.41e-5\
    --batch_size 1\
    --seq_length 128\
    --gradient_accumulation_steps 1\
    --load_in_4bit True\
    --load_in_8bit False\
    --use_peft True\
    --output_dir '/home/uework/AiWeb/pwz/github/model/luachat/test'\
    --peft_lora_r 64\
    --peft_lora_alpha 16\
    --logging_steps 1\
    --num_train_epochs 10\
    --save_strategy 'epoch'\
    --save_total_limit 10\
    --dataset_text_field 'content'\


