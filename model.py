from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn.util import batched_index_select
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class BertForChID(BertPreTrainedModel):

    # _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # if config.is_decoder:
        #     logger.warning(
        #         "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
        #         "bi-directional self-attention."
        #     )

        self.bert = BertModel(config, add_pooling_layer=False)
        # self.bert_for_idiom = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.alpha1 = 0.5
        self.alpha2 = 0.5
        self.loss_func = torch.nn.CrossEntropyLoss()
        self._model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._model_device = "cpu"
        self.W_blank = nn.Linear(768, 500)
        self.W_blank.requires_grad_= True
        self.W_idiom = nn.Linear(768, 500)
        self.W_idiom.requires_grad_= True
        self.dropout = nn.Dropout(0.0)
        
        # self.bert_for_idiom.requires_grad_ = False

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
        position: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labels_syn: Optional[torch.Tensor] = None,
        candidates: Optional[torch.Tensor] = None,
        synonyms: Optional[torch.Tensor] = None,
        synonyms_mask: Optional[torch.Tensor] = None,
        synonyms_len: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_synonyms = False,
        is_train: Optional[bool] = True
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels: torch.LongTensor of shape `(batch_size, )`
        candidates: torch.LongTensor of shape `(batch_size, num_choices, 4)`
        candidate_mask: torch.BooleanTensor of shape `(batch_size, seq_len)`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        len = input_ids.shape[1]
        
        # 上下文的CLS embedding
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
        )
        sequence_output = outputs_blank[0][:, 0, :].repeat(1, 7).reshape(-1, 7, 768)
        sequence_output = torch.relu(self.W_blank(sequence_output))
        sequence_output = F.normalize(sequence_output, dim=-1)
        sequence_output = self.dropout(sequence_output)
        
        # candidate embedding
        with torch.no_grad():
            # inputs_cand = self.replace_mask(input_ids, candidates, position, candidate_mask).view(input_ids.shape[0], -1, len).permute(1,0,2).to(self._model_device)
            inputs_cand = candidates.to(self._model_device)
            outputs_cand = []
            for cand in inputs_cand:
                outputs_cand.append(self.bert(
                    cand,
                    # attention_mask=attention_mask.to(self._model_device),
                    # token_type_ids=token_type_ids.to(self._model_device),
                    attention_mask=torch.ones_like(cand).to(self._model_device),
                    token_type_ids=torch.zeros_like(cand).to(self._model_device),
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                ).last_hidden_state
                )
            outputs_cand = torch.stack(outputs_cand, dim=0)
            candidate_output = outputs_cand[:,:,0,:]
            # candidate_output = outputs_cand.permute(1,0,2,3)
            # candidate_output = outputs_cand[torch.tensor(list(range(outputs_cand.shape[0]))).to(self._model_device),:,position.to(self._model_device),:]
            
        candidate_output = torch.relu(self.W_idiom(candidate_output))
        candidate_output = F.normalize(candidate_output, dim=-1)
        candidate_output = self.dropout(candidate_output)

        # 和candidate的相似度
        '''
        print(sequence_output.shape)
        print(candidate_output.shape)
        '''
        sim1 = torch.sum(sequence_output*candidate_output, dim=-1)
        score1 = sim1
        # score1 = torch.softmax(sim1, dim=-1)
        loss1 = self.loss_func(score1, labels.to(self._model_device))
        # print(torch.sum(self.W_idiom.weight.grad))
        
        if not is_train or not use_synonyms:
            return loss1, score1, labels
        
         # 近义词embedding
        with torch.no_grad():
            inputs_syn = self.replace_mask(input_ids, synonyms, position, candidate_mask).view(input_ids.shape[0], -1, len).permute(1,0,2).to(self._model_device)
            outputs_syn = []
            for syn in inputs_syn:
                outputs_syn.append(self.bert(
                    syn,
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
                )
            outputs_syn = torch.stack(outputs_syn, dim=0)
            synonyms_output = outputs_syn[:,:,0,:]
            # synonyms_output = outputs_syn.permute(1,0,2,3)
            # synonyms_output = outputs_syn[torch.tensor(list(range(outputs_syn.shape[0]))).to(self._model_device),:,position.to(self._model_device),:]
        
            synonyms_output = torch.relu(self.W_idiom(synonyms_output))
            synonyms_output = F.normalize(synonyms_output, dim=-1)
            synonyms_output = self.dropout(synonyms_output)
            
        # 和近义词的相似度
        '''
        print(synonyms_output.shape)
        print(sequence_output.shape)
        '''
        sim2 = torch.sum(sequence_output*synonyms_output, dim=-1)
        #print(synonyms_mask)
        sim2 = sim2 * synonyms_mask.to(self._model_device)
        score2 = torch.softmax(sim2, dim=-1)
        loss2 = self.loss_func(score2, labels_syn.to(self._model_device))
        
        loss = loss1 + loss2
        
        return loss, torch.cat((score1, score2), dim=0), torch.cat((labels, labels_syn), dim=0)

        
    def replace_mask(self, input_ids, candidates, position, candidate_mask):
        len = input_ids.shape[1]
        #print(candidates)
        #print("position")
        #print(position)
        candidate_expand = torch.stack([torch.cat([torch.zeros(pos, dtype=torch.int64).to(self._model_device), c, torch.zeros(len-pos-4, dtype=torch.int64).to(self._model_device)]) for cand,pos in zip(candidates, position) for c in cand], dim=0)
        input_cand = torch.where(candidate_mask.repeat(1,7).view(-1,len), candidate_expand, input_ids.repeat(1,7).view(-1,len))
        return input_cand
            