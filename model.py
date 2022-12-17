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

        # if config.is_decoder:
        #     logger.warning(
        #         "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
        #         "bi-directional self-attention."
        #     )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self._model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.syn_loss_rate = 0.5

        # 全连接层，用于训练
        self.input_len = 768 # 网络输入层的大小 21128 768
        self.times = 2 # 网络中间层的放大倍数
        self.hidden1 = torch.nn.Linear(self.input_len, self.times*self.input_len)
        # self.hidden2 = torch.nn.Linear(self.times*self.input_len, self.times*self.input_len)
        # self.hidden3 = torch.nn.Linear(self.times*self.input_len, self.times*self.input_len)
        self.output = torch.nn.Linear(self.times*self.input_len, self.input_len)
        
        self.relu = torch.nn.ReLU()
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

        # with torch.no_grad():
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
            
            # MLP
        sequence_output = self.relu(self.hidden1(sequence_output))
        sequence_output = self.relu(self.output(sequence_output))
        
        prediction_scores = self.cls(sequence_output) # (Batch_size, Seq_len, Vocab_size)


        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss() 
            mask_prediction_scores = torch.masked_select(prediction_scores, candidate_mask.unsqueeze(-1)).reshape(-1, prediction_scores.shape[-1], 1) # (Batch_size x 4, Vocab_size, 1)

            ################# MLP #################
            # mask_prediction_scores = torch.masked_select(prediction_scores, candidate_mask.unsqueeze(-1)).reshape(-1, prediction_scores.shape[-1])  # [batch_size*4, 21128]
            # # print("[INFO] mask_prediction_scores.shape: ", mask_prediction_scores.shape)
            # # 以下参数与初始化参数配合用来训练网络
            # mask_prediction_scores = F.gelu(self.hidden1(mask_prediction_scores))
            # # mask_prediction_scores = F.gelu(self.hidden2(mask_prediction_scores))
            # # mask_prediction_scores = F.gelu(self.hidden3(mask_prediction_scores))
            # mask_prediction_scores = self.output(mask_prediction_scores)
            # # 恢复成(Batch_size*4, Vocab_size, 1)的形式
            # mask_prediction_scores = mask_prediction_scores.unsqueeze(-1)  # [batch_size*4, 21128, 1]
            # # print("[INFO] mask_prediction_scores.shape: ", mask_prediction_scores.shape)
            #######################################
            
            # candidate
            candidate_indices = candidates.transpose(-1, -2).reshape(-1, candidates.shape[1]).contiguous() # (Batch_size x 4, num_choices)
            candidate_logits = batched_index_select(mask_prediction_scores, candidate_indices).squeeze(-1).reshape(prediction_scores.shape[0], 4, -1).transpose(-1, -2) # (Batch_size, num_choices, 4)

            candidate_labels = labels.reshape(labels.shape[0], 1).repeat(1, 4) # (Batch_size, 4)
            final_scores = torch.sum(F.log_softmax(candidate_logits, dim=-2), dim=-1) # (Batch_size, num_choices)
            masked_lm_loss = loss_fct(candidate_logits, candidate_labels)
            
            # synonyms
            if use_synonyms:
                synonyms_indices = synonyms.transpose(-1, -2).reshape(-1, synonyms.shape[1]).contiguous() # (Batch_size x 4, num_choices)
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