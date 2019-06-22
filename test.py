import os
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from models.bert import *
from utils.data_loader import QNLILoader


def test(args):
    # =============== Setup GPU ================== #
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device('cuda', args.gpu_id)

    # =============== Setup Logger ================== #
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    with open(os.path.join(args.log_path, 'test.log'), 'w') as f:
        pass
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.log_path, 'test.log'),
        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S',
        level = logging.INFO
    )

    # =============== Setup Seeds ================== #
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # =============== Setup Data ================== #
    test_loader = QNLILoader(args, 'test')
    lbl_map = {v: k for k, v in test_loader.lbl_map.items()}

    import pickle
    if not os.path.exists('data/.cache/features_t.pkl'):
        test_features = test_loader.get_features()
        with open('data/.cache/features_t.pkl', 'wb') as f:
            pickle.dump(test_features, f)
    else:
        with open('data/.cache/features_t.pkl', 'rb') as f:
            test_features = pickle.load(f)

    all_input_ids = torch.tensor([f['input_ids'] for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

    # =============== Setup Models ================== #
    model_map = {'rawbert': RawBertCls, 'bertcnnmlp': BertClsCNNMLP, 'bertmlp': BertClsMLP, 'bertlstm': BertClsLSTM}
    with open('model_list.txt', 'r') as f:
        model_list = f.readlines()
    models = []
    for m in model_list:
        if m[0] == '#':
            continue
        model_type, model_path = m.split('|')
        model = model_map[model_type](args.bert_model, args)
        if model_path[-1] == '\n':
            model.load_state_dict(torch.load(model_path[:-1]))
        else:
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        models.append(model)
        print('load model from: ', model_path[:-1])

    # =============== Testing ================== #
    preds = []
    f = open('QNLI.tsv', 'w')
    f.write('index\tprediction\n')

    for step, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids = batch

        max_probs_list = []
        prob_list = []
        pred_list = []
        with torch.no_grad():
            for model in models:
                logit = model(input_ids, segment_ids, input_mask, labels=None)
                prob = F.softmax(logit, dim=-1)

                pred = prob.max(1, keepdim=True)[1].cpu().numpy()
                prob_list.append(prob.cpu().detach().numpy()[0])
                max_probs_list.append(prob.max(1, keepdim=True)[0].cpu().detach().numpy()[0][0])
                pred_list.append(pred)

        max_probs_list = np.array(max_probs_list)

        # ensemble
        if args.vote == 'hard':
            max_pred = pred_list[np.argmax(max_probs_list)][0][0]
        elif args.vote == 'soft':
            prob_sum = np.zeros((2))
            for sub_prob in prob_list:
                prob_sum += sub_prob
            max_pred = np.argmax(prob_sum)

        f.write(str(step) + '\t' + lbl_map[max_pred] + '\n')
            
    print('test ended')
    f.close()