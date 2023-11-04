# Fine-tuning LuaCoder for chat-based applications

## Version

**luacoder toolkit 0.0.1**
- data processing for chat interaction and code generation
- model training using deepspeed with chat and code generation model
- evaluation with code and chat inference

## File Structure

```
luacoder/
    set-up.sh       # some short cut for quick setup
    README.md       # this file
    eval/           # eval model
        ......
    deepspeed/      # train model
        chat/       # chat generation model
           ......
        pretrain/   # pretain training & code generation training
            ......
        multi/      # all in one train pack (not complete)
            ......
        main/       # simple train scripts for quick start up
            ......
        RL/         # PEFT train and RLHF training
            ......
    data/           # training data processing
        ......

```
## Getting started

To run the data processing, training and evaluation script, first create a Python virtual environment using e.g. Conda:

```shell
conda create -n train python=3.10 -y && conda activate train
```

Next, install PyTorch v2.0.1. Since this is hardware-dependent, we direct you to the [PyTorch Installation Page](https://pytorch.org/get-started/previous-versions/#v1131) for this step. Next, install the rest of the project dependencies:

```shell
pip install torch==2.0.1 torchvision torchaudio
pip install -r deepspeed/chat/requirements.txt
```

Each sections (data processing, training, evaluation) may have different requirements and they may have conflict with each other, use `pip install -r requirements.txt` under each folder for specific package requirements.

## Prepare your dataset

Using data folder
- for lua dataset generation refering to lua.ipynb, lua files generation using json documents and dialogue, generated dataset can be used for code generation
- for json processing refering to json.ipynb, can be used to transform markdown document to json format
- for dialogue processing refering to dialogue.ipynb, generated dataset can be used for instructional chat generation

### Instruction chat generation 
For training and inference, we use _dialogue templates_ to format each message in a conversation. For example, a typical dialogue between a human user and AI assistant takes the form:

```json
{
    "messages": [
        {
            "content": "Is it possible to imagine a society without law?", 
            "role": "user"},
        {
            "content": "It is difficult to imagine a society that is able to be maintained without any semblance of Law.",
            "role": "assistant",
        },
        {
            "content": "It seems like you consider the absence of law equal to the absence of anything that could guide the behaviour of the individual.",
            "role": "user",
        },
        {
            "content": "You are correct that there are other factors that can guide behavior in a society and play a role in shaping individuals' behavior and interactions with each other. However, even in societies where these factors are present, laws still serve an important role in maintaining social order and resolving conflicts.",
            "role": "assistant",
        }
    ]
}
```

Make sure you convert your dataset according to this schema, in particular you need to include a `messages` column like the above. You can adjust the model, dataset, and hyperparamters in the `config.yaml` file.

### Code generation or pretrain model

For training and inference, we use below to format each message in a conversation. For example, a dataset contain `content` takes the form:
```json
{
    "content": "You are correct that there are other factors that can guide behavior in a society and play a role in shaping individuals' behavior and interactions with each other. However, even in societies where these factors are present, laws still serve an important role in maintaining social order and resolving conflicts."
}
```
Make sure you convert your dataset according to this schema, in particular you need to include a `content` column like the above. You can adjust the model, dataset, and hyperparamters in the `config.yaml` file.

## Launch training

Using deepspeed folder
- into main, simple example for training scripts quick start up
- into chat folder for chat based instruction model
- into pretrain for specific programing language code generation model
- into multi, still developping for overall training to avoid duplication (not complete)
- into RL folder for PEFT training and RLHF training (not complete)

We use DeepSpeed ZeRO-3 to shard the model and optimizer across 4 x A100 (40GB) GPUs. To fine-tune run:

```
conda activate train
cd deepspeed/chat
TRANSFORMERS_VERBOSITY=info torchrun --nproc_per_node=4 train.py config.yaml --deepspeed=deepspeed_z3_config_bf16.json
```

## Generate samples and eval


Currently, the evaluation is based on calculation the match accuracy of the api detected in the generated text and the api documents. The higher accuracy means the generated text follow the guidelines of the api documents to generate code or other data.

To generate report, you can modify the config in eval/eval.py, which will automatic evaluate all checkpoints in the output folders to identify which fit best for the requirements.

There are three data source needed to be configure
- base_folder: which is the checkpoints root folder.
- api_path: json files format data of api documents, json can be generated in Prepare your dataset section.
- test_data_path: csv format data that contains `prompt` colomn.

Using eval folder
- for chat evaluation, modify config `report_type` to `markdown`, as generated data are `markdown`.
- for code evaluation, modify config `report_type` to `code`, as generated data are lua `code`.

### run evaluation
```
# set environment
conda create -n eval python=3.10 -y && conda activate eval
cd eval
pip install -r requirements.txt
# execute evaluation
python eval.py
```



To generate a few coding examples from your model, using eval_code.ipynb and eval_chat.ipynb playground.