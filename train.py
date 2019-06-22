import os
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from models.bert import *
from utils.data_loader import QNLILoader


def train(args):
    # =============== Setup GPU ================== #
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device('cuda', args.gpu_id)

    # =============== Setup Logger ================== #
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    with open(os.path.join(args.log_path, 'train.log'), 'w') as f:
        pass
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.log_path, 'train.log'),
        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt = '%m/%d/%Y %H:%M:%S',
        level = logging.INFO
    )

    # =============== Setup Seeds ================== #
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # =============== Setup Data ================== #
    train_loader = QNLILoader(args, 'train')
    dev_loader = QNLILoader(args, 'dev')
    
    import pickle
    if not os.path.exists('data/.cache/features.pkl'):
        print('generating features for %s ' % train_loader.mode)
        train_features = train_loader.get_features()
        dev_features = dev_loader.get_features()
        with open('data/.cache/features.pkl', 'wb') as f:
            pickle.dump([train_features, dev_features], f)
    else:
        print('loading features for %s ' % train_loader.mode)
        with open('data/.cache/features.pkl', 'rb') as f:
            train_features, dev_features = pickle.load(f)

    num_samples = len(train_features)
    num_labels = 2
    num_train_optimization_steps = num_samples // args.batch_size * args.num_epochs

    # transfer features from list to torch.Tensor
    all_input_ids = torch.tensor([f['input_ids'] for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f['lbl'] for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    all_input_ids_eval = torch.tensor([f['input_ids'] for f in dev_features], dtype=torch.long)
    all_input_mask_eval = torch.tensor([f['input_mask'] for f in dev_features], dtype=torch.long)
    all_segment_ids_eval = torch.tensor([f['segment_ids'] for f in dev_features], dtype=torch.long)
    all_label_ids_eval = torch.tensor([f['lbl'] for f in dev_features], dtype=torch.long)

    dev_data = TensorDataset(all_input_ids_eval, all_input_mask_eval, all_segment_ids_eval, all_label_ids_eval)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.batch_size)

    # =============== Setup Model ================== #
    model = BertClsLSTM(args.bert_model, args)
    model.to(device)

    # =============== Setup Optimizer ================== #
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion, t_total=num_train_optimization_steps)

    # =============== Training ================== #
    all_tr_loss = 0
    best_acc = 0

    for ep in trange(int(args.num_epochs), desc="Epoch"):
        tr_loss = 0

        # train an epoch
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids, segment_ids, input_mask, labels=None)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            tr_loss += loss.item()
        
        all_tr_loss += tr_loss    
        logger.info('[epoch: %d ] loss: %0.6f | avg loss %0.6f' % \
            (ep, tr_loss / num_samples, all_tr_loss / ((ep + 1) * num_samples)))

        # evaluate on development set
        model.eval()
        preds = []

        for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
        preds = preds[0]
        preds = np.argmax(preds, axis=1)

        acc = (preds == all_label_ids_eval.numpy()).mean()
        print(acc)
        logger.info('[epoch: %d ] accuracy: %0.4f' % (ep, acc))

        # save checkpoint
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(args.model_path, 'bertcls_lr%f_batchsize%d_ep%d_acc%0.4f.pth' % (args.lr, args.batch_size, ep, acc))
            torch.save(model.state_dict(), save_path)
            logger.info('Save better model in epoch %d' % ep)
        save_path = os.path.join(args.model_path, 'bertcls_lr%f_batchsize%d_last.pth' % (args.lr, args.batch_size))
        torch.save(model.state_dict(), save_path)