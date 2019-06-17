import torch.nn as nn
import torch.nn.functional as F

# from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
from pytorch_pretrained_bert.modeling import BertModel, BertLayerNorm


# class BertLayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         super(BertLayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias


class RawBertCls(nn.Module):
    def __init__(self, bert_model):
        super(RawBertCls, self).__init__()
        self.backbone = BertModel.from_pretrained('data/.cache/bert-base-uncased.tar.gz')
        for para in self.backbone.parameters():
            para.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.backbone(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            print('bertlayernorm')
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



class BertClsCNNMLP(nn.Module):
    def __init__(self, bert_model):
        super(BertClsCNNMLP, self).__init__()
        self.backbone = BertModel.from_pretrained('data/.cache/bert-base-uncased.tar.gz')
        for para in self.backbone.parameters():
            para.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv1d(1, 64, 5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(64, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(256, 16, 1, stride=1, padding=0)

        self.fc1 = nn.Linear(16 * 192, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 2)
        # self.classifier = nn.Linear(768, 2)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.backbone(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        x = F.relu(F.max_pool1d(self.conv1(pooled_output.unsqueeze(1)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)

        return logits


class BertClsMLP(nn.Module):
    def __init__(self, bert_model):
        super(BertClsMLP, self).__init__()
        self.backbone = BertModel.from_pretrained('data/.cache/bert-base-uncased.tar.gz')
        # for para in self.backbone.parameters():
        #     para.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.backbone(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        probs = self.classifier(pooled_output)

        return probs
