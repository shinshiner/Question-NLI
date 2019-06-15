import os
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from models.bert import BertCls
from utils.data_loader import QNLILoader


def test(args):
    # =============== Setup GPU ================== #
    if args.gpu and args.gpu_id != -1 and torch.cuda.is_available():
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
    print(lbl_map)

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

    model = BertCls(args.bert_model)
    model.to(device)
    print('loading model from : %s' % os.path.join(args.model_path, args.model_name))
    model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_name)))

    # =============== Testing ================== #
    preds = []
    for step, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids = batch

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # preds.append(logits.detach().cpu().numpy())

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    preds = np.argmax(preds, axis=1)

        # np_pred = np.zeros((0, 2))
        # for pred in preds:
        #     np_pred = np.append(np_pred, pred, axis=0)
        # np_pred = np.argmax(np_pred, axis=1)
        # eval_loss /= num_eval_steps

    with open('submission.tsv', 'w') as f: 
        f.write('index\tprediction\n')
        for i, pred in enumerate(preds):
            f.write(str(i) + '\t' + lbl_map[pred] + '\n')
    print('test ended')