from transformers import BertModel
import torch.nn as nn
from transformers import BertPreTrainedModel


class BertMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids, tag_embeds, attention_mask=None, labels=None):

        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, tag_embeds=tag_embeds)

        hidden_state, pooled_output = outputs[:2]

        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)

        try:
            loss_fc = nn.BCEWithLogitsLoss()
            loss = loss_fc(logits, labels.float())
        except:
            loss = None

        return loss, logits



