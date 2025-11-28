export OPENCV_IO_ENABLE_OPENEXR=1
export TOKENIZERS_PARALLELISM=false

export TASK_NAME="depth" # or normal

# paths
export CORE_PREDICTOR_MODEL_PATH="weights/lotus-2_core_predictor_$TASK_NAME.safetensors"
export DETAIL_SHARPENER_MODEL_PATH="weights/lotus-2_detail_sharpener_$TASK_NAME.safetensors"
export LCM_MODEL_PATH="weights/lotus-2_lcm_$TASK_NAME.safetensors"

export INPUT_DIR="assets/in-the-wild_examples"
export OUTPUT_DIR="outputs/infer/"

# configs
export NUM_INFERENCE_STEPS=10

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --num_inference_steps=$NUM_INFERENCE_STEPS \
    --seed="0" \
    --task_name=$TASK_NAME \
    --core_predictor_model_path=$CORE_PREDICTOR_MODEL_PATH \
    --detail_sharpener_model_path=$DETAIL_SHARPENER_MODEL_PATH \
    --lcm_model_path=$LCM_MODEL_PATH