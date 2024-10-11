RUN_NAME=fp16_training
CKPT_DIR="ckpt"

echo "Running training: $RUN_NAME"
echo "CKPT Dir: $CKPT_DIR"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name=c4 \
    --dataset_config_name=realnewslike \
    --max_new_tokens=200 \
    --min_prompt_tokens=200 \
    --max_input_len=500 \
    --min_generations=1000 \
    --input_truncation_strategy=completion_length \
    --input_filtering_strategy=prompt_and_completion_length \
    --output_filtering_strategy=max_new_tokens \
    --run_name=$RUN_NAME \
    --wandb=False \
    --verbose=False \
    --batch_size=8 \
    --ckpt_dir=$CKPT_DIR \
    --lr=0.0001 \
    --layer_gamma=2 \
    --layer_delta=2 \
    --init_val_gamma=0.25 \
    --init_val_delta=2.0 \
    --log_z_score=False \
    --z_score_factor=1.0 \
    --optimizer=Adam \
    --method=Pareto \
    --epochs=2 

