import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    RobertaForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
)


class BertModel(nn.Module):
    def __init__(self, model_name, requires_grad=True):
        super(BertModel, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.requires_grad = requires_grad
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
        )[:2]
        if labels is None:
            return logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, in_size=1, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(in_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(
            *[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.output_layer(x)
        return torch.sigmoid(x) * 2
