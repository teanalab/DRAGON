# import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class CERModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', dropout_prob=0.1):
        super(CERModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)  # Output single pruning score
        self.lamda = .5
        self.cosine = nn.CosineSimilarity()
    def forward(self, input_ids, attention_mask, token_type_ids, Enc_Q, Enc_C, relevance):
        # BERT forward pass
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        cls_token_output = outputs[1]  # CLS token encoding
        # Apply dropout and linear layer for pruning score
        dropout_output = self.dropout(cls_token_output)
        score = self.linear(dropout_output)
        
        cosine_sim = self.cosine(Enc_Q.pooler_output, Enc_C.pooler_output)
        

        return cosine_sim, score

        # return pruning_score, prob_pruning_score