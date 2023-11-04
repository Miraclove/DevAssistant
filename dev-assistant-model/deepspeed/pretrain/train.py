#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The BigCode & HuggingFace Inc. teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Script to instruction fine-tune causal language models on a Hub dataset

Adapted from huggingface/transformers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""

import logging
import math
import os
import random
import sys
from itertools import chain

import datasets
import torch
import transformers
from config import DataArguments, ModelArguments, TrainingArguments
from datasets import load_dataset,load_from_disk,IterableDataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          default_data_collator, set_seed)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from utils import StarChatArgumentParser, hf_login

logger = logging.getLogger(__name__)
# Import wandb
import wandb


def main():
    parser = StarChatArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a YAML file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    # parse command line args and yaml file
    elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
    # parse command line args only
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    if training_args.local_rank == 0:  # only on main process
        # Initialize wandb run
        # log in wandb
        wandb.login()
        # set project name
        wandb.init(project="luacoder", name=model_args.wandb_run_name)


    ###########################
    # Detecting last checkpoint
    ###########################
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ###############
    # Load datasets
    ###############
    logger.info("*** Load datasets ***")
    raw_datasets = load_from_disk(data_args.dataset_name)
    logger.info("*** Load datasets done ***")
    # logger.info(
    #     f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    # )
    # with training_args.main_process_first(desc="Log a few random samples from the raw training set"):
    #     for index in random.sample(range(len(raw_datasets["train"])), 3):
    #         logger.info(f"Sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['messages']}")

    #####################################
    # Load tokenizer and process datasets
    #####################################

    logger.info("*** Load tokenizer ***")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
    )
    logger.info("*** Process datasets ***")



    # with training_args.main_process_first(desc="Log a few random samples from the training set"):
    #     for index in random.sample(range(len(raw_datasets["train"])), 3):
    #         logger.info(f"Sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['content']}")

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
    logger.info("*** Tokenize function ***")
    def tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=data_args.block_size,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == data_args.block_size:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    logger.info("*** Tokenize datasets ***")
    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]
    logger.info("*** Tokenize datasets done ***")
    logger.info("train_dataset length",len(train_dataset))
    logger.info("eval_dataset length",len(eval_dataset))

    

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    # accelerate + quantized
    if model_args.quantized:
        from transformers import BitsAndBytesConfig
        logger.info("*** use accelerate + quantized ***")
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            quantization_config=bnb_config,
            use_cache=False if training_args.gradient_checkpointing else True,
        )
    else:
        # deepspeed or accelerate without quantized
        logger.info("*** use deepspeed ***")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
        )

    ########################
    # Initialize the Trainer
    ########################

    from transformers import DataCollatorForLanguageModeling
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator defaults to DataCollatorWithPadding, so we change it
        # since we've already chunked our corpus
        data_collator=data_collator,
    )

    ###############
    # Training loop
    ###############
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    


if __name__ == "__main__":
    main()
