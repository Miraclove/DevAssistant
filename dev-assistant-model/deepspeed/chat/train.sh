# deeepspeed + accelerate distributed parallel
TRANSFORMERS_VERBOSITY=info torchrun --nproc_per_node=4 train.py config.yaml --deepspeed=deepspeed_z3_config_bf16.json

# accelerate quantisized
# CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_VERBOSITY=info python train.py config.yaml

# deepspeed
# deepspeed --include localhost:1,2 train.py config.yaml --deepspeed=deepspeed_z3_config_bf16.json