#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for ChID.
"""


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union
import random
from tqdm import tqdm, trange
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import datasets
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
from datasets import load_dataset
from model import BertForChID

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from decoder_my import *



logger = logging.getLogger(__name__)
VOCAB_SIZE = 21128


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    use_synonyms: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If passed, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to the maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class DataCollatorForChID:
    """
    Data collator that will dynamically pad the inputs.
    Candidate masks will be computed to indicate which tokens are candidates.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        try:
            label_name = "label" if "label" in features[0].keys() else "labels"
            labels = [feature.pop(label_name) for feature in features]
        except:
            label_name = "label" if "label" in features.keys() else "labels"
            labels = features.pop(label_name)

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )


        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        # Compute candidate masks
        batch["candidate_mask"] = batch["input_ids"] == self.tokenizer.mask_token_id
        return batch

def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.train_file is not None or data_args.validation_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Downloading and loading the chid dataset from the hub. This code is not supposed to be executed in.
        raw_datasets = load_dataset(
            "YuAnthony/chid",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # model = BertForChID.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    model = make_model(VOCAB_SIZE, VOCAB_SIZE, N=6)

    label_column_name = "labels"
    idiom_tag = '#idiom#'

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Preprocessing the datasets.

    # We only consider one idiom per instance in the dataset, a sentence containing multiple idioms will be split into multiple instances.
    # The idiom tag of each instance will be replaced with 4 [MASK] tokens.
    def preprocess_function_resize(examples):
        return_dic = {}
        return_dic_keys = ['candidates', 'synonyms', 'synonyms_len', 'content', 'labels', 'labels_syn']
        for k in return_dic_keys:
            return_dic[k] = []

        for i in range(len(examples['content'])):
            idx = -1
            text = examples['content'][i]
            for j in range(examples['realCount'][i]):
                idx = text.find(idiom_tag, idx+1)
                return_dic['content'].append(text[:idx] + tokenizer.mask_token*4 + text[idx+len(idiom_tag):])
                
                # candidates
                return_dic['candidates'].append(examples['candidates'][i][j])
                for k, candidate in enumerate(examples['candidates'][i][j]):
                    if candidate == examples['groundTruth'][i][j]:
                        return_dic['labels'].append(k)
                        break
                    
                # synonyms
                examples['synonyms'][i][j] = [examples['groundTruth'][i][j]] + examples['synonyms'][i][j]
                examples['synonyms'][i][j] = [exa for exa in examples['synonyms'][i][j] if len(exa)==4][:7]
                random.shuffle(examples['synonyms'][i][j])
                syn_len = len(examples['synonyms'][i][j])
                examples['synonyms'][i][j] = (examples['synonyms'][i][j] + ["M"*4]*7)[:7]
                
                return_dic['synonyms'].append(examples['synonyms'][i][j])
                return_dic['synonyms_len'].append(syn_len)
                for k, candidate in enumerate(examples['synonyms'][i][j]):
                    if candidate == examples['groundTruth'][i][j]:
                        return_dic['labels_syn'].append(k)
                        break
        return return_dic

    # tokenize all instances
    def preprocess_function_tokenize(examples):
        first_sentences = examples['content']
        labels = examples["labels"]
        labels_syn = examples["labels_syn"]
        # truncate the first sentences.
        for i, sentence in enumerate(first_sentences):
            if len(sentence) <= 500:
                continue
            if sentence.find(tokenizer.mask_token*4) > len(sentence) // 2:
                first_sentences[i] = sentence[-500:]
            else:
                first_sentences[i] = sentence[:500]
                
        tokenized_examples = tokenizer(
            first_sentences,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True,
        )
        # candidates
        tokenized_examples["labels"] = labels
        tokenized_candidates = [[tokenizer.convert_tokens_to_ids(list(candidate)) for candidate in candidates]for candidates in examples['candidates']]
        tokenized_examples["candidates"] = tokenized_candidates
        
        # synonyms
        tokenized_examples["labels_syn"] = labels_syn
        tokenized_synonyms = [[tokenizer.convert_tokens_to_ids(list(synonym)) for synonym in synonyms]for synonyms in examples['synonyms']]
        tokenized_examples["synonyms"] = tokenized_synonyms
        
        # Data collator
        data_collator = DataCollatorForChID(tokenizer=tokenizer,  pad_to_multiple_of=8 if training_args.fp16 else None)
    
        data_set = data_collator(tokenized_examples.data).data
        tokenized_examples['candidate_mask'] = data_set['candidate_mask'].tolist()
        return tokenized_examples
    
    def generate_dataset(dataset):
        dataset = dataset.map(
            preprocess_function_resize,
            batched=True,
            remove_columns=["groundTruth", "realCount", "explaination", "exp embedding"],
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        dataset = dataset.map(
            preprocess_function_tokenize,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache
        )
        data_set = {key:torch.tensor(dataset[:][key]) for key in tqdm(dataset[0].keys()) if key != 'content'}
        dataset = TensorDataset(data_set['input_ids'], data_set['token_type_ids'], data_set['attention_mask'], data_set['candidates'], data_set['synonyms'], data_set['synonyms_len'], data_set['labels'], data_set['labels_syn'], data_set['candidate_mask'])
        data_loader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
        return data_loader
    
    keys = ['input_ids', 'token_type_ids', 'attention_mask', 'candidates', 'synonyms', 'synonyms_len', 'labels', 'labels_syn', 'candidate_mask']
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_data_loader = generate_dataset(train_dataset)
            
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_data_loader = generate_dataset(eval_dataset)
        test_dataset = raw_datasets["test"]
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            test_data_loader = generate_dataset(test_dataset)
    



    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions.detach().cpu().numpy(), axis=1)
        return {"accuracy": (preds == label_ids.cpu().numpy()).astype(np.float32).mean().item()}

    model.to(model._model_device)
    optimizer = torch.optim.Adam(params = model.parameters(),lr=training_args.learning_rate)
    # lr_scheduler = LambdaLR(
    #     optimizer=optimizer,
    #     lr_lambda=lambda step: rate(
    #         step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
    #     ),
    # )
    lr_scheduler = None
    criterion = LabelSmoothing(size=7, padding_idx=0, smoothing=0.0)

    # Training
    if training_args.do_train:
        model.train()
        max_acc = 0.
        # TODO
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        
        for epoch in range(training_args.num_train_epochs):
            # total_loss = 0.
            # total_correct = 0.
            # total_num = 0.
            # step = 0
            model.train()
            run_epoch(train_data_loader, 
                    model, 
                    SimpleLossCompute(model.generator, criterion),
                    optimizer,
                    lr_scheduler,
                    mode="train",
                    )
            
            # for batch in train_data_loader:
            #     step += 1
            #     optimizer.zero_grad()
                
            #     batch = [b.to(model._model_device) for b in batch]
                
            #     input = dict(zip(keys,batch))
            #     output = model(**input, use_synonyms = data_args.use_synonyms)
                
            #     output.loss.backward()
            #     optimizer.step()
                
            #     if data_args.use_synonyms:
            #         labels = torch.cat((input['labels'], input['labels_syn']), dim=0)
            #     else:
            #         labels = input['labels']
            #     metrics = compute_metrics((output.logits, labels))
            #     total_loss += output.loss
            #     total_correct += metrics["accuracy"]
            #     # total_num += len(batch[-2])
            #     total_num += 1
                
                # rate = step / len(train_data_loader)
                # a = "*" * int(rate * 50)
                # b = "." * int((1 - rate) * 50)
                # print("\rtrain loss: {:^3.0f}%[{}->{}] loss: {:.4f}  accuracy: {:.4f}".format(int(rate*100), a, b, output.loss, metrics["accuracy"]*1.0), end="")

                
            # logger.info("\nepoch: {:.0f} loss: {:.4f}  accuracy: {:.4f}".format(epoch+1, total_loss, total_correct/total_num))

            # total_loss = 0.
            # total_correct = 0.
            # total_num = 0.
            # step = 0
            
            # model.eval()
            # with torch.no_grad():
            #     for batch in eval_data_loader:
                    
            #         batch = [b.to(model._model_device) for b in batch]
                    
            #         input = dict(zip(keys,batch))
            #         output = model(**input)
                    
            #         metrics = compute_metrics((output.logits, batch[-2]))
            #         total_loss += output.loss
            #         total_correct += metrics["accuracy"]
            #         total_num += 1
                    
            # logger.info("\neval loss: {:.4f}  accuracy: {:.4f}".format(total_loss, total_correct/total_num))
            # model.train()
        
        
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        total_loss = 0.
        total_correct = 0.
        total_num = 0.
        step = 0
        
        model.eval()
        with torch.no_grad():
            for batch in test_data_loader:
                
                batch = [b.to(model._model_device) for b in batch]
                
                input = dict(zip(keys,batch))
                output = model(**input)
                
                metrics = compute_metrics((output.logits, batch[-2]))
                total_loss += output.loss
                total_correct += metrics["accuracy"]
                total_num += 1
                
        logger.info("\ntest loss: {:.4f}  accuracy: {:.4f}".format(total_loss, total_correct/total_num))

    

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()