export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ALGO=1 # 1代表la，0代表fa

torchrun --master_port=2031 --nproc_per_node=16 \
    sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "examples/case1" \
    --prompt "An old-fashioned European village with thatched roofs on the houses." \
    --i2v-stability \
    --infer-steps 2 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --save-path ./results \
    --ulysses-degree 8 \
    --ring-degree 2 \
    --use_attentioncache \
    --vae-parallel