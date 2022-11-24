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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import datasets
import numpy as np
import torch
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
from datasets import load_dataset
from model import BertForChID

import transformers
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
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
from transformers import AdamW, get_linear_schedule_with_warmup



logger = logging.getLogger(__name__)
_model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        default=False,
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
        label_name = "label" if "label" in features.keys() else "labels"
        label_syn_name = "label_syn" if "label_syn" in features.keys() else "labels_syn"
        labels = features.pop(label_name)
        labels_syn = features.pop(label_syn_name)

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )


        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        batch["labels_syn"] = torch.tensor(labels_syn, dtype=torch.int64)
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
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
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
    model = BertForChID.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    ).to(_model_device)
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer
    #                 if not any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.01, 'lr': 1e-5},
    #     {'params': [p for n, p in param_optimizer
    #                 if any(nd in n for nd in no_decay) and 'bert' in n], 'weight_decay': 0.0, 'lr': 1e-5},
    #     {'params': [p for n, p in param_optimizer
    #                 if not any(nd in n for nd in no_decay) and 'bert' not in n], 'weight_decay': 0.01, 'lr': training_args.learning_rate},
    #     {'params': [p for n, p in param_optimizer
    #                 if any(nd in n for nd in no_decay) and 'bert' not in n], 'weight_decay': 0.0, 'lr': training_args.learning_rate}
    # ]
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, training_args.learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), training_args.learning_rate)
    optimizer = torch.optim.Adam(params = model.parameters(),lr=1e-4)

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
        return_dic_keys = ['candidates', 'content', 'synonyms', 'synonyms_mask', 'explaination', 'exp embedding', 'labels', 'labels_syn']
        for k in return_dic_keys:
            return_dic[k] = []

        for i in range(len(examples['content'])):
            idx = -1
            text = examples['content'][i]
            for j in range(examples['realCount'][i]):
                return_dic['candidates'].append(examples['candidates'][i][j])
                idx = text.find(idiom_tag, idx+1)
                return_dic['content'].append(text[:idx] + tokenizer.mask_token*4 + text[idx+len(idiom_tag):])
                examples['synonyms'][i][j] = [examples['groundTruth'][i][j]] + examples['synonyms'][i][j]
                # examples['synonyms'][i][j] = ([exa for exa in examples['synonyms'][i][j] if len(exa)==4] \
                #                             + [[0]*4]*7)[:7]
                # random.shuffle(examples['synonyms'][i][j])
                examples['synonyms'][i][j] = [exa for exa in examples['synonyms'][i][j] if len(exa)==4][:7]
                random.shuffle(examples['synonyms'][i][j])
                
                syn_len = len(examples['synonyms'][i][j])
                return_dic['synonyms_mask'].append([1]*syn_len+[0]*(7-syn_len))
                # examples['synonyms_mask'][i][j] = len(examples['synonyms'][i][j][:7])
                examples['synonyms'][i][j] = (examples['synonyms'][i][j] + [[tokenizer.mask_token]*4]*7)[:7]
                return_dic['synonyms'].append(examples['synonyms'][i][j])
                return_dic['explaination'].append(examples['explaination'][i][j])
                return_dic['exp embedding'].append(examples['exp embedding'][i][j])
                for k, candidate in enumerate(examples['candidates'][i][j]):
                    if candidate == examples['groundTruth'][i][j]:
                        return_dic['labels'].append(k)
                        break
                for k, candidate in enumerate(examples['synonyms'][i][j]):
                    if candidate == examples['groundTruth'][i][j]:
                        return_dic['labels_syn'].append(k)
                        break
        return return_dic

    # tokenize all instances
    def preprocess_function_tokenize(examples):
        first_sentences = examples['content']
        labels = examples[label_column_name]
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
        tokenized_examples["labels"] = examples['labels']
        # tokenized_candidates = [[tokenizer.convert_tokens_to_ids(list(candidate)) for candidate in candidates]for candidates in examples['candidates']]
        tokenized_candidates = [[tokenizer(candidate).input_ids for candidate in candidates]for candidates in examples['candidates']]
        tokenized_examples["candidates"] = tokenized_candidates
        
        tokenized_examples["labels_syn"] = examples['labels_syn']
        # examples["synonyms"] = [(syn+[[0,0,0,0]]*7)[:7] for syn in examples["synonyms"]]
        tokenized_synonyms = [[tokenizer.convert_tokens_to_ids(list(synonym)) for synonym in synonyms]for synonyms in examples['synonyms']]
        tokenized_examples["synonyms"] = tokenized_synonyms
        tokenized_examples["synonyms_mask"] = examples['synonyms_mask']
        
        tokenized_examples["position"] = [l.index(tokenizer.mask_token_id) for l in tokenized_examples["input_ids"]]
        
        # Data collator
        data_collator = (
            default_data_collator
            if data_args.pad_to_max_length
            else DataCollatorForChID(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
        )
    
        data_set = data_collator(tokenized_examples.data).data
        keys = list(data_set.keys())
        data_set = TensorDataset(data_set['input_ids'], data_set['token_type_ids'], data_set['attention_mask'], data_set['candidates'], data_set['synonyms'], data_set['synonyms_mask'], data_set['position'], data_set['labels'], data_set['labels_syn'], data_set['candidate_mask'], )
        return data_set, keys

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
            
        logger.info("train dataset map pre-processing")
        train_dataset = {k:[d[k] for d in train_dataset] for k in train_dataset.column_names}
        train_dataset = preprocess_function_resize(train_dataset)
        train_dataset, keys = preprocess_function_tokenize(train_dataset)
        train_data_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=False)
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info("validation dataset map pre-processing")
        eval_dataset = {k:[d[k] for d in eval_dataset] for k in eval_dataset.column_names}
        eval_dataset = preprocess_function_resize(eval_dataset)
        eval_dataset, keys = preprocess_function_tokenize(eval_dataset)
        eval_data_loader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
        
        test_dataset = raw_datasets["test"]
        logger.info("test dataset map pre-processing")
        test_dataset = {k:[d[k] for d in test_dataset] for k in test_dataset.column_names}
        test_dataset = preprocess_function_resize(test_dataset)
        test_dataset, keys = preprocess_function_tokenize(test_dataset)
        test_data_loader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)


    # Metric
    def compute_metrics(predictions, label_ids):
        predictions, label_ids = predictions.detach().cpu().numpy(), label_ids.detach().cpu().numpy()
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).sum().item()}


    # Training
    num_train_optimization_steps = len(train_data_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * training_args.warmup_ratio), num_train_optimization_steps)
    
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
            total_loss = 0.
            total_correct = 0.
            total_num = 0.
            step = 0
            for batch in train_data_loader:
                step += 1
                
                batch = [b.to(model._model_device) for b in batch]
                
                input = dict(zip(keys,batch))
                loss, logits, labels = model(is_train=True, **input)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                metrics = compute_metrics(logits, labels)
                total_loss += loss
                total_correct += metrics["accuracy"]
                total_num += len(labels)
                
                rate = step / len(train_data_loader)
                # a = "*" * int(rate * 50)
                # b = "." * int((1 - rate) * 50)
                # print("\rtrain loss: {:^3.0f}%[{}->{}] loss: {:.4f}  accuracy: {:.4f}".format(int(rate*100), a, b, loss, metrics["accuracy"]*1.0/len(labels)), end="")

                
            logger.info("\nepoch: {:.0f} loss: {:.4f}  accuracy: {:.4f}".format(epoch+1, total_loss, total_correct/total_num))
            
            # eval
            # model.eval()
            total_correct = 0.
            total_num = 0.
            with torch.no_grad():
                for batch in eval_data_loader:
                    
                    input = dict(zip(keys,batch))
                    loss, logits, labels = model(is_train=False, **input)
                    
                    metrics = compute_metrics(logits, labels)
                    total_correct += metrics["accuracy"]
                    total_num += len(labels)
                
            logger.info("eval accuracy: {:.4f}".format(total_correct/total_num))
            if max_acc < total_correct/total_num:
                save_trained_model(training_args.output_dir, model)
            # model.train()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        model = BertForChID.from_pretrained(
            training_args.output_dir,
            from_tf=bool(".ckpt" in training_args.output_dir),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        ).to(_model_device)
        # model.eval()
        total_correct = 0.
        total_num = 0.
        for batch in test_data_loader:
            
            input = dict(zip(keys,batch))
            loss, logits, labels = model(is_train=False, **input)
            
            metrics = compute_metrics(logits, labels)
            total_correct += metrics["accuracy"]
            total_num += len(labels)
            
        logger.info("test accuracy: {:.4f}".format(total_correct/total_num))

    # kwargs = dict(
    #     finetuned_from=model_args.model_name_or_path,
    #     tasks="multiple-choice",
    #     dataset_tags="swag",
    #     dataset_args="regular",
    #     dataset="SWAG",
    #     language="en",
    # )

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)
    
def save_trained_model(output_dir, model):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger.info('Saving model to %s'%output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()