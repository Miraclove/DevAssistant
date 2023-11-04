# deeepspeed + accelerate distributed parallel
TRANSFORMERS_VERBOSITY=info torchrun --nproc_per_node=4 train.py config.yaml --deepspeed=deepspeed_z3_config_bf16.json

# accelerate quantisized
# TRANSFORMERS_VERBOSITY=info python train.py config.yaml