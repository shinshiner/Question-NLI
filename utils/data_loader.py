import os

from pytorch_pretrained_bert.tokenization import BertTokenizer


class QNLILoader():
    def __init__(self, args, mode):
        self.args = args
        self.data_path = args.data_path
        self.mode = mode
        self.num_samples = None
        self.lbl_map = {'not_entailment': 1, 'entailment': 0}

    # examples --> features
    def get_features(self):
        examples = self.get_examples()

        tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=self.args.lower_case)
        features = []

        for i, example in enumerate(examples):
            token_q = tokenizer.tokenize(example['q'])
            token_a = tokenizer.tokenize(example['a'])

            # control length
            while True:
                num_length = len(token_q) + len(token_a)
                if num_length <= (self.args.max_seq_len - 3):
                    break
                if len(token_q) > len(token_a):
                    token_q.pop()
                else:
                    token_a.pop()

            # fix format
            token = ['[CLS]'] + token_q + ['[SEP]'] + token_a + ['[SEP]']
            segment_ids = [0] * (len(token_q) + 2) + [1] * (len(token_a) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(token)
            input_mask = [1] * len(input_ids)

            # padding
            padding = [0] * (self.args.max_seq_len - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            lbl = self.lbl_map[example['lbl']]

            features.append({'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'lbl': lbl})

        return features

    # raw text --> examples
    def get_examples(self):
        # read .tsv file
        if self.mode == 'test':
            with open(os.path.join(self.data_path, '%s.tsv' % self.mode), 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    tmp = line.strip()
                    idx, q, a = tmp.split('\t')
                    lines[i] = [idx, q, a]
        else:
            with open(os.path.join(self.data_path, '%s.tsv' % self.mode), 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    tmp = line.strip()
                    idx, q, a, lbl = tmp.split('\t')
                    lines[i] = [idx, q, a, lbl]

        # parse examples
        examples = []
        lines = lines[1:]
        self.num_samples = len(lines)

        for i, line in enumerate(lines):
            guid = '%s-%s' % (self.mode, line[0])
            q, a = line[1], line[2]
            if self.mode != 'test':
                lbl = line[-1]
            else:
                lbl = 'entailment'

            examples.append({'guid': guid, 'q': q, 'a': a, 'lbl': lbl})

        return examples
        
    def __len__(self):
        return self.num_samples