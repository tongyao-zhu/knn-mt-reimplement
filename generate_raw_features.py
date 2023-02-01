# Code adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py
# !/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os
import random

import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
)
from transformers.utils.versions import require_version

logger = get_logger(__name__)
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/translation/requirements.txt",
)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    # Parse the arguments
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=[f"train[:{args.percentage}%]"],
        )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, output_hidden_states=True
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    logger.info(f"Raw datasets: {raw_datasets}")
    column_names = raw_datasets[0].column_names

    # Get the language codes for input/target.
    source_lang = args.source_lang.split("_")[0]
    target_lang = args.target_lang.split("_")[0]

    padding = "max_length" if args.pad_to_max_length else False

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        prefix = ""
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=args.max_source_length, padding=padding, truncation=True
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets[0].map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Running tokenizer on dataset with {len(raw_datasets[0])} entries",
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    label_pad_token_id = -100
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_batch_size,
    )

    # Prepare everything with our `accelerator`.
    model, train_dataloader = accelerator.prepare(
        model,
        train_dataloader,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config[
                "lr_scheduler_type"
            ].value
            accelerator.init_trackers("translation_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_batch_size * accelerator.num_processes

    logger.info("***** Running feature generation (Inference on training set) *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel & distributed) = {total_batch_size}"
    )
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(len(train_dataset) // total_batch_size),
        disable=not accelerator.is_local_main_process,
    )
    starting_epoch = 0

    def helper_print(name, tensor):
        # a function to help with debugging
        print("Tensor {} has shape {}".format(name, tensor.shape))

    # set the model to evaluation mode, as we are only doing inference (one forward pass on the training set)
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            assert (
                len(outputs.decoder_hidden_states) == 7
            )  # should have 1 embedding layer + 6 transformers layers
            assert batch.decoder_input_ids.size(1) == batch.labels.size(1)
            keys = outputs.decoder_hidden_states[-1].flatten(0, 1)  # [B*T, hidden_size]
            values = batch.labels.flatten()

            # remove the values whose next token id is 1, which means padding
            mask = values > 1
            keys = keys[mask]
            values = values[mask]
            assert len(values) == len(keys) == mask.sum()
            torch.save(keys, os.path.join(args.save_path, f"{step}.pt"))
            torch.save(values, os.path.join(args.save_path, f"{step}_values.pt"))
            progress_bar.update(1)  # one step finished


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate raw feature tensors for building the datastore"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        type=str,
        help="directory name to save the generated features, e.g. saved_gen",
    )
    parser.add_argument(
        "--percentage",
        type=int,
        required=False,
        default=1,
        help="The percentage of dataset to extract features. This is for testing purpose",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )

    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help=(
            "Whether to pad all samples to model maximum sentence "
            "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default=None,
        help="Source language id for translation.",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default=None,
        help="Target language id for translation.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    args = parser.parse_args()

    # Sanity checks

    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a task name or a training/validation file.")

    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in [
            "csv",
            "json",
        ], "`train_file` should be a csv or a json file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in [
            "csv",
            "json",
        ], "`validation_file` should be a csv or a json file."

    return args


if __name__ == "__main__":
    main()
