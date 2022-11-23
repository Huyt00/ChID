'''
lyg: 
    通过BertOnlyMLMHead类获取BERT对每个位置上下文关系的预测结果,
    这个向量的大小是 [batch_size, word_count, vocabulary_size] = [32, 482, 21128],
    其中vocabulary_size指BERT经过tokenizer后词典的大小,
    这里的结果是词典中每一个词在句中第i位置上的可能性大小.
    不训练BERT网络, 添加n个全连接层对BertOnlyMLMHead得到的[MASK]位置上的预测结果进行映射并训练该网络,
    期待正确成语的4位预测概率和最大, 此时做softmax()和CrossEntropyLoss()配合即可.
    同时考虑candidates和synonyms的loss.
    self.input_len用来调整网络输入大小, 该值等于vocabulary_size
    self.times用来指定映射放大倍数(由于GPU内存大小此时选用1)
    epoch = 20
    learning_rate = 1e-4
'''
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from allennlp.nn.util import batched_index_select
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from typing import Optional, Tuple, Union

class BertForChID(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.predict = BertOnlyMLMHead(config)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self._model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 全连接层，用于训练
        self.input_len = 21128 # 网络输入层的大小
        self.times = 1 # 网络中间层的放大倍数
        # self.hidden1 = torch.nn.Linear(self.input_len, self.times*self.input_len)
        # self.hidden2 = torch.nn.Linear(self.times*self.input_len, self.times*self.input_len)
        # self.hidden3 = torch.nn.Linear(self.times*self.input_len, self.times*self.input_len)
        # self.output = torch.nn.Linear(self.times*self.input_len, self.input_len)

        # 设置candidates和synonyms不同的loss rate
        self.loss1_rate = 1
        self.loss2_rate = 0.1

        self.post_init()

    def get_output_embeddings(self):
        return self.predict.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.predict.predictions.decoder = new_embeddings

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
        candidates: Optional[torch.Tensor] = None,
        labels_syn: Optional[torch.Tensor] = None,
        synonyms: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_train: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels: torch.LongTensor of shape `(batch_size, )`
        candidates: torch.LongTensor of shape `(batch_size, num_choices, 4)`
        candidate_mask: torch.BooleanTensor of shape `(batch_size, seq_len)`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        len = input_ids.shape[1]

        # print("candidate_mask type:", type(candidate_mask))
        # print("candidate_mask.shape: ", candidate_mask.shape)
        # candidate_mask type: <class 'torch.Tensor'>
        # candidate_mask.shape:  torch.Size([32, 482])

        # print("candidates type:", type(candidates))
        # print("candidates.shape: ", candidates.shape)
        # candidates type: <class 'torch.Tensor'>
        # candidates.shape:  torch.Size([32, 7, 4])

        with torch.no_grad():
            outputs_blank = self.bert(
                input_ids.to(self._model_device),
                attention_mask=attention_mask.to(self._model_device),
                token_type_ids=token_type_ids.to(self._model_device),
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        # print(outputs_blank.shape)
        # torch.Size([32, 482, 768])
        # print(position.shape)
        # torch.Size([32])
        outputs_blank = self.predict(outputs_blank)  # get prediction score, 单词表中每个词的输出概率
        # print("outputs_blank.shape cls: ", outputs_blank.shape)
        # outputs_blank.shape cls:  torch.Size([32, 482, 21128])

        pos_mask = self.getPosMask(input_ids, position).to(self._model_device)
        # print("**pos_mask.shape: ", pos_mask.shape)
        # torch.Size([32, 482, 768])
        all_prediction_score = torch.masked_select(outputs_blank, pos_mask).reshape(input_ids.shape[0], 4, 21128)
        
        # 以下参数与初始化参数配合用来训练网络
        # all_prediction_score = F.gelu(self.hidden1(all_prediction_score))
        # all_prediction_score = F.gelu(self.hidden2(all_prediction_score))
        # all_prediction_score = F.gelu(self.hidden3(all_prediction_score))
        # all_prediction_score = self.output(all_prediction_score)

        all_prediction_score = all_prediction_score.unsqueeze(-1).reshape(input_ids.shape[0]*4, -1, 1)

        # 测试view()报错的问题
        # all_prediction_score = torch.masked_select(outputs_blank, candidate_mask.unsqueeze(-1)).reshape(-1, outputs_blank.shape[-1], 1) # (Batch_size x 4, Vocab_size, 1)
        
        ################ candidates ################
        # indices要加.contiguous(), 否则batched_index_select()会报view()函数的奇怪错误
        candidate_indices = candidates.transpose(-1, -2).reshape(-1, candidates.shape[1]).contiguous().to(self._model_device) # (Batch_size x 4, num_choices)
        candidate_logits = batched_index_select(all_prediction_score, candidate_indices).squeeze(-1).reshape(input_ids.shape[0], 4, -1).transpose(-1, -2) # (Batch_size, num_choices, 4)

        # labels用老师的方法是有问题的, 要改写
        # labels = labels.reshape(labels.shape[0], 1).repeat(1, 4) # (Batch_size, 4)
        # print("labels.shape: ", labels.shape)
        # labels.shape:  torch.Size([1])
        score1 = torch.sum(F.log_softmax(candidate_logits, dim=-2), dim=-1) # (Batch_size, num_choices)
        loss1 = self.loss_func(score1, labels.to(self._model_device))

        ############### synonyms #################
        synonyms_indices = synonyms.transpose(-1, -2).reshape(-1, synonyms.shape[1]).contiguous().to(self._model_device)
        synonyms_logits = batched_index_select(all_prediction_score, synonyms_indices).squeeze(-1).reshape(input_ids.shape[0], 4, -1).transpose(-1, -2)
        score2 = torch.sum(F.log_softmax(synonyms_logits, dim=-2), dim=-1)
        loss2 = self.loss_func(score2, labels_syn.to(self._model_device))

        ################ final loss ################
        loss = self.loss1_rate*loss1 + self.loss2_rate*loss2

        if not is_train:
            return loss1, score1, labels
        return loss, torch.cat((score1, score2), dim=0), torch.cat((labels, labels_syn), dim=0)

    def getPosMask(self, input_ids, position):
        len = input_ids.shape[1]
        # 要求mask是BoolTensor类型, 此时mask是[32, 482]的，需要扩展维度
        num = 21128
        mask = torch.stack([torch.cat([torch.zeros([pos,num], dtype=torch.int64), torch.ones(4,num), torch.zeros([len-pos-4,num], dtype=torch.int64)]) for pos in position], dim=0) > 0
        return mask
        
    def replace_mask(self, input_ids, candidates, position, candidate_mask):
        # 在position位置填上candidates词
        len = input_ids.shape[1]
        candidate_expand = torch.stack([torch.cat([torch.zeros(pos, dtype=torch.int64), c, torch.zeros(len-pos-4, dtype=torch.int64)]) for cand,pos in zip(candidates, position) for c in cand], dim=0)
        
        input_cand = torch.where(candidate_mask.repeat(1,7).view(-1,len), candidate_expand, input_ids.repeat(1,7).view(-1,len))
        return input_cand