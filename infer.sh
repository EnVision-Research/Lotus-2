export TASK_NAME="depth" # or normal

export INPUT_DIR="assets/in-the-wild_examples"
export OUTPUT_DIR="outputs/infer/"

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --seed="0" \
    --task_name=$TASK_NAME