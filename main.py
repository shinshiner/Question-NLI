import os
import sys
import random

import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from train import train


def init_args():
    parser = argparse.ArgumentParser()

    # basic setting
    parser.add_argument('--data_path', default='data', type=str, help='path to the data files')
    parser.add_argument('--model_path', default='trained_models', type=str, help='path to the trained models')
    parser.add_argument('--log_path', default='logs', type=str, help='path to logs')
    parser.add_argument('--mode', default='train', type=str, help='train or eval')
    parser.add_argument('--lower_case', action='store_true', help='whether to uncase the sequence')
    parser.add_argument('--gpu', action='store_true', help='whether to use gpu')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id')
    parser.add_argument('--random_seed', default=42, type=int, help='random seed')

    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, help='bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased')

    # training settings
    parser.add_argument('--max_seq_len', default=128,  type=int, help='maximum length of input sequence')
    parser.add_argument('--num_epochs', default=50,  type=int, help='training epochs')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Proportion of linear learning rate warmup')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = init_args()

    # setup directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.mode == 'train':
        print('======= training mode ========')
        train(args)