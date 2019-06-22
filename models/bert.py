import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def init_conv(m):
    weight_shape = list(m.weight.data.size())
    fan_in = np.prod(weight_shape[1:4])
    fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
    w_bound = np.sqrt(6. / (fan_in + fan_out))
    m.weight.data.uniform_(-w_bound, w_bound)
    m.bias.data.fill_(0)


class RawBertCls(nn.Module):
    def __init__(self, bert_model, args):
        super(RawBertCls, self).__init__()
        self.backbone = BertModel.from_pretrained('data/.cache/bert-base-uncased.tar.gz')

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.backbone(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        probs = self.classifier(pooled_output)

        return probs


class BertClsCNNMLP(nn.Module):
    def __init__(self, bert_model, args):
        super(BertClsCNNMLP, self).__init__()
        self.backbone = BertModel.from_pretrained('data/.cache/bert-base-uncased.tar.gz')

        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv1d(1, 64, 5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(64, 256, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(256, 16, 1, stride=1, padding=0)

        self.fc1 = nn.Linear(16 * 192, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.backbone(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        x = F.relu(F.max_pool1d(self.conv1(pooled_output.unsqueeze(1)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool1d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        probs = self.fc4(x)

        return probs


class BertClsMLP(nn.Module):
    def __init__(self, bert_model, args):
        super(BertClsMLP, self).__init__()
        self.backbone = BertModel.from_pretrained('data/.cache/bert-base-uncased.tar.gz')

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.backbone(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        probs = self.classifier(pooled_output)

        return probs


class BertClsLSTM(nn.Module):
    def __init__(self, bert_model, args):
        super(BertClsLSTM, self).__init__()
        self.args = args
        self.hidden_size = 768
        self.lstm_hidden_size = 256
        self.cnn_feature_size = self.args.max_seq_len * 2
        self.backbone = BertModel.from_pretrained('data/.cache/bert-base-uncased.tar.gz')

        self.conv1 = nn.Conv2d(1, 256, (7, self.hidden_size), stride = 1, padding = (3, 0))
        self.conv2 = nn.Conv1d(256, 64, 5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(64, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, 16, 1, stride=1, padding=0)

        self.lstm = nn.LSTMCell(self.hidden_size, self.lstm_hidden_size)

        self.fc1 = nn.Linear(self.lstm_hidden_size + self.cnn_feature_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)

        self.fc1.weight.data = norm_col_init(self.fc1.weight.data, 1.0)
        self.fc1.bias.data.fill_(0)

        self.fc2.weight.data = norm_col_init(self.fc2.weight.data, 1.0)
        self.fc2.bias.data.fill_(0)

        self.fc3.weight.data = norm_col_init(self.fc3.weight.data, 1.0)
        self.fc3.bias.data.fill_(0)

        init_conv(self.conv1)
        init_conv(self.conv2)
        init_conv(self.conv3)
        init_conv(self.conv4)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        all_encoder_layers, pooled_output = self.backbone(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        all_encoder_layers = torch.sum(torch.stack(all_encoder_layers[8:]), 0) # all_encoder_layers - [batch_size * max_seq_len * hidden_size]

        bs = all_encoder_layers.size()[0]

        all_encoder_layers_lstm_format = all_encoder_layers.permute(1, 0, 2)
        cx = torch.zeros(bs, self.lstm_hidden_size).cuda()
        hx = torch.zeros(bs, self.lstm_hidden_size).cuda()
        hxs = torch.zeros((all_encoder_layers_lstm_format.size()[0], bs, self.lstm_hidden_size)).cuda()
        x0 = torch.zeros(bs, 1, self.args.max_seq_len, all_encoder_layers_lstm_format.size(1)).cuda()

        for i in range(all_encoder_layers_lstm_format.size()[0]):
            hx, cx = self.lstm(all_encoder_layers_lstm_format[i], (hx, cx))
            hxs[i] = hx
        
        hx_mean = torch.mean(hxs, 0, True).squeeze(0) # [1 * batch_size * lstm_hidden_size]

        x0 = F.relu(F.max_pool1d(self.conv1(all_encoder_layers.unsqueeze(1)).squeeze(3), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool1d(self.conv2(x0), kernel_size=2, stride=2))
        x0 = F.relu(F.max_pool1d(self.conv3(x0), kernel_size=2, stride=2))
        x0 = F.relu(self.conv4(x0))
        
        x0 = x0.view(x0.size(0), -1)
        
        x = torch.cat((hx_mean, x0), 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x
