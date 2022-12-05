import time
import copy
import math
import numpy as np
import dill
import random
from typing import Optional, Union
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
from allennlp.nn.util import batched_index_select

from transformers import BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

CLS = 101


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

def get_dataset(tokenizer, max_seq_length, model_args, data_args, training_args, idiom_tag='#idiom#'):
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
        tokenized_examples["candidate_pos"] = [[l.index(tokenizer.mask_token_id)] for l in tokenized_examples["input_ids"]]
        
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

    def generate_dataset(dataset, shuffle=False):
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
        dataset = TensorDataset(data_set['input_ids'], data_set['token_type_ids'], data_set['attention_mask'], data_set['candidates'], data_set['synonyms'], data_set['synonyms_len'], data_set['labels'], data_set['labels_syn'], data_set['candidate_mask'], data_set['candidate_pos'])
        data_loader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, shuffle=shuffle)
        return data_loader

    train_data_loader = None
    eval_data_loader = None
    test_data_loader = None
    if data_args.reload_dataset:
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
        
        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_data_loader = generate_dataset(train_dataset, shuffle=True)
                
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

        with open(data_args.train_file[:-4]+'pkl','wb') as f:
            dill.dump(train_data_loader, f)
        with open(data_args.validation_file[:-4]+'pkl','wb') as f:
            dill.dump(eval_data_loader, f)
        with open(data_args.test_file[:-4]+'pkl','wb') as f:
            dill.dump(test_data_loader, f)
    else:
        with open(data_args.train_file[:-4]+'pkl','rb') as f:
            train_data_loader = dill.load(f)
        with open(data_args.validation_file[:-4]+'pkl','rb') as f:
            eval_data_loader = dill.load(f)
        with open(data_args.test_file[:-4]+'pkl','rb') as f:
            test_data_loader = dill.load(f)

    return train_data_loader, eval_data_loader, test_data_loader


# Metric
def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions.detach().cpu().numpy(), axis=1)
    return {"accuracy": (preds == label_ids.cpu().numpy()).astype(np.float32).mean().item()}

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, outputs, candidates, candidate_mask, labels=None, vocab_len=4):
        prediction_scores = self.generator(outputs)
        norm = outputs.size(0)*vocab_len
        
        loss_fct = CrossEntropyLoss() 
        mask_prediction_scores = torch.masked_select(prediction_scores, candidate_mask.unsqueeze(-1)).reshape(-1, prediction_scores.shape[-1], 1) # (Batch_size x 4, Vocab_size, 1)
        # mask_prediction_scores = prediction_scores.reshape(-1, prediction_scores.shape[-1], 1) # (Batch_size x 4, Vocab_size, 1)
       
        candidate_indices = candidates.transpose(-1, -2).reshape(-1, candidates.shape[1]) # (Batch_size x 4, num_choices)
        candidate_logits = batched_index_select(mask_prediction_scores, candidate_indices).squeeze(-1).reshape(prediction_scores.shape[0], vocab_len, -1) # (Batch_size, 4, num_choices)
        final_scores = torch.sum(F.log_softmax(candidate_logits, dim=-1), dim=-2) # (Batch_size, num_choices)
        
        if labels is None:
            return final_scores

        candidate_labels = labels.reshape(labels.shape[0], 1).repeat(1, vocab_len) # (Batch_size, 4)
        masked_lm_loss = loss_fct(candidate_logits.transpose(-2, -1), candidate_labels)
        
        return masked_lm_loss * norm, masked_lm_loss, final_scores
        
        sloss = (
            self.criterion(
                candidate_logits.contiguous().view(-1, candidate_logits.size(-1)), candidate_labels.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss, final_scores

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None

class DummyScheduler:
    def step(self):
        None

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class BertForChID(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # if config.is_decoder:
        #     logger.warning(
        #         "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
        #         "bi-directional self-attention."
        #     )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self._model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MaskedLMOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output="'paris'",
    #     expected_loss=0.88,
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labels_syn: Optional[torch.Tensor] = None,
        candidates: Optional[torch.Tensor] = None,
        candidate_pos: Optional[torch.Tensor] = None,
        synonyms: Optional[torch.Tensor] = None,
        synonyms_len: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_synonyms = False
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels: torch.LongTensor of shape `(batch_size, )`
        candidates: torch.LongTensor of shape `(batch_size, num_choices, 4)`
        candidate_mask: torch.BooleanTensor of shape `(batch_size, seq_len)`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output) # (Batch_size, Seq_len, Vocab_size)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss() 
            mask_prediction_scores = torch.masked_select(prediction_scores, candidate_mask.unsqueeze(-1)).reshape(-1, prediction_scores.shape[-1], 1) # (Batch_size x 4, Vocab_size, 1)
            
            # candidate
            candidate_indices = candidates.transpose(-1, -2).reshape(-1, candidates.shape[1]) # (Batch_size x 4, num_choices)
            candidate_logits = batched_index_select(mask_prediction_scores, candidate_indices).squeeze(-1).reshape(prediction_scores.shape[0], 4, -1).transpose(-1, -2) # (Batch_size, num_choices, 4)

            candidate_labels = labels.reshape(labels.shape[0], 1).repeat(1, 4) # (Batch_size, 4)
            final_scores = torch.sum(F.log_softmax(candidate_logits, dim=-2), dim=-1) # (Batch_size, num_choices)
            masked_lm_loss = loss_fct(candidate_logits, candidate_labels)
            
            # synonyms
            if use_synonyms:
                synonyms_indices = synonyms.transpose(-1, -2).reshape(-1, synonyms.shape[1]) # (Batch_size x 4, num_choices)
                synonyms_logits = batched_index_select(mask_prediction_scores, synonyms_indices).squeeze(-1).reshape(prediction_scores.shape[0], 4, -1).transpose(-1, -2) # (Batch_size, num_choices, 4)

                synonyms_labels = labels_syn.reshape(labels_syn.shape[0], 1).repeat(1, 4) # (Batch_size, 4)
                synonyms_final_scores = torch.sum(F.log_softmax(synonyms_logits, dim=-2), dim=-1) # (Batch_size, num_choices)
                masked_lm_loss += self.syn_loss_rate*loss_fct(synonyms_logits, synonyms_labels)
                final_scores = torch.cat((final_scores, synonyms_final_scores), dim=0)


        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=final_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = encoder.cls
        self._model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, src, tgt, src_mask, tgt_mask, input):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask, input), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask, input):
        return self.encoder(**input)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return memory
        tgt_emb = torch.masked_select(memory, tgt.unsqueeze(-1)).reshape(tgt.size(0), 4, -1)
        return tgt_emb
        src_mask_ = src_mask * (1 - tgt.type_as(src_mask))
        return self.decoder(tgt_emb, memory, src_mask_.unsqueeze(1), tgt_mask)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab, config):
        super(Generator, self).__init__()
        # self.proj = nn.Linear(d_model, vocab)
        self. proj = BertOnlyMLMHead(config)

    def forward(self, x):
        # return log_softmax(self.proj(x), dim=-1)
        return self.proj(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            # query, key, value, mask=mask.repeat(1,self.h,query.size(-2),1), dropout=self.dropout
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def make_std_mask(tgt, pad=4):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
        tgt_mask.data
    )
    return tgt_mask

def make_model(
    model_args, config, N=6, d_model=768, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        BertForChID.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        ),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)
    
    # EncoderDecoder.encoder = bert
    return model

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_correct = 0.
    total_num = 0.
    
    keys = ['input_ids', 'token_type_ids', 'attention_mask', 'candidates', 'synonyms', \
            'synonyms_len', 'labels', 'labels_syn', 'candidate_mask', 'candidate_pos']
    
    for i, batch in enumerate(data_iter):
        
        batch = [b.to(model._model_device) for b in batch]
        input = dict(zip(keys,batch))
        batch_size = batch[0].size(0)
        
        src = input['input_ids']
        src_mask = input['attention_mask']
        # candidates = input['candidates'][torch.tensor(list(range(src.size(0)))),input['labels'],:]
        # pad = torch.zeros(batch_size, 1).fill_(CLS).type_as(candidates)
        # tgt = torch.cat((pad, candidates), dim=-1)[:,:-1]
        # tgt = input['candidate_pos'].repeat(1, 4) + torch.range(0, 3).unsqueeze(0).repeat(batch_size, 1).type_as(input['candidate_pos'])
        tgt = input['candidate_mask']
        tgt_mask = make_std_mask(torch.zeros(batch_size, 4).type_as(src))
        # can_mask = input['candidate_mask']
        out = model(
            src, tgt, src_mask, tgt_mask, input
        )
        # tgt_y = input['candidates']
        # labels = input['labels']
        # # if mode == "train" or mode == "train+log":
        # loss_node, loss, logits = loss_compute(out, tgt_y, input['candidate_mask'], labels=labels)
        # # else:
        # #     loss, loss_node, logits = loss_compute(out[:, 0, :], tgt_y[:, :, 0], labels=labels, vocab_len=1)
        loss = out.loss
        logits = out.logits
        labels = input['labels']
        metrics = compute_metrics((logits, labels))
        # loss_node = loss_node / accum_iter
        
        if mode == "train" or mode == "train+log":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        total_loss += loss
        total_correct += metrics["accuracy"]
        total_num += 1
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            # lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Loss: %6.2f "
                    + "| Step / Sec: %7.1f | Accuracy: %6.2f"
                )
                % (i, loss, total_num / elapsed, metrics["accuracy"])
            )
            start = time.time()
        # del loss
        # del loss_node
    return total_loss, total_correct/total_num

def greedy_decode(model, src, src_mask, candidates, groundTruth, logit_func, max_len=4, start_symbol=CLS):
    batch_size = src.size(0)
    memory = model.encode(src, src_mask)
    ys = torch.zeros(batch_size, 1).fill_(start_symbol).type_as(src.data)
    prob_all = []
    for i in range(max_len):
        ys_mask = make_std_mask(ys, pad=ys.size(1))
        out = model.decode(
            memory, src_mask, ys, ys_mask
        )
        prob = logit_func(out[:, -1], candidates[:,:,i], vocab_len=1)
        prob_all.append(prob)
        _, next_word_idx = torch.max(prob, dim=1)
        next_word = batched_index_select(candidates, next_word_idx.unsqueeze(-1))[:,:,i]
        # next_word = batched_index_select(candidates, groundTruth.unsqueeze(-1))[:,:,i]
        ys = torch.cat([ys, next_word], dim=1)
    prob_all = torch.stack(prob_all, dim=1)
    final_scores = torch.sum(F.log_softmax(prob_all, dim=-1), dim=-2) # (Batch_size, num_choices)
    return ys, final_scores

def run_test(
    data_iter,
    model,
    loss_compute,
    optimizer=None,
    scheduler=None,
    mode="eval",
):
    """Train a single epoch"""
    total_correct = 0.
    total_num = 0.
    
    keys = ['input_ids', 'token_type_ids', 'attention_mask', 'candidates', 'synonyms', \
            'synonyms_len', 'labels', 'labels_syn', 'candidate_mask']
    
    for i, batch in enumerate(data_iter):
        
        batch = [b.to(model._model_device) for b in batch]
        input = dict(zip(keys,batch))
        
        src = input['input_ids']
        src_mask = input['attention_mask']
        tgt_y = input['candidates']
        
        _, logits = greedy_decode(model, src, src_mask, tgt_y, input['labels'], logit_func=loss_compute)
        
        metrics = compute_metrics((logits, input['labels']))

        total_correct += metrics["accuracy"]
        total_num += 1
    return None, total_correct/total_num
