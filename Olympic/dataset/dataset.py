import codecs
from torch.utils.data import Dataset
import numpy as np

# train_file = 'data/train.txt'
# test_file = 'data/test.txt'

def pad_sent(sent, pad, max_length):
    sent_arr = sent.split()
    sent_arr.extend([pad for i in range(max_length)])
    return ' '.join(sent_arr[:max_length]).strip()


class TrainDataBert(Dataset):
    def __init__(self, train_file, s1_length, s2_length, max_length, tokenizer):
        self.s1_length = s1_length
        self.s2_length = s2_length
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.pairs = []

        with codecs.open(train_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                info = line.strip().split('  [SEP]  ')
                self.pairs.append((info[0], info[1], int(info[2])))

    def __getitem__(self, index):
        data = self.pairs[index]
        if self.max_length == 0:
            Q1 = pad_sent(data[0], '[PAD]', self.s1_length)
            Q2 = pad_sent(data[1], '[PAD]', self.s2_length)
            max_length = self.s1_length + self.s2_length + 3
        else:
            Q1 = data[0]
            Q2 = data[1]
            max_length = self.max_length

        tokenzied_dict = self.tokenizer.encode_plus(text=Q1,
                                                    text_pair=Q2,
                                                    max_length=max_length,
                                                    pad_to_max_length=True)
        return np.array(tokenzied_dict['input_ids']), np.array(tokenzied_dict['token_type_ids']), np.array(
            tokenzied_dict['attention_mask']), np.array([data[2]])

    def __len__(self):
        return len(self.pairs)

class EvalDataBert(Dataset):
    def __init__(self, test_file, s1_length, s2_length, max_length, tokenizer):
        self.s1_length = s1_length
        self.s2_length = s2_length
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.pairs = []

        with codecs.open(test_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                info = line.strip().split('  [SEP]  ')
                Q1 = info[0]
                Q2 = info[1]
                if len(Q1) > self.s1_length:
                    Q1 = Q1[:self.s1_length]
                if len(Q2) > self.s2_length:
                    Q2 = Q2[:self.s2_length]
                self.pairs.append((Q1, Q2, int(info[2])))

    def __getitem__(self, index):
        data = self.pairs[index]
        if self.max_length == 0:
            Q1 = pad_sent(data[0], '[PAD]', self.s1_length)
            Q2 = pad_sent(data[1], '[PAD]', self.s2_length)
            max_length = self.s1_length + self.s2_length + 3
        else:
            Q1 = data[0]
            Q2 = data[1]
            max_length = self.max_length

        tokenzied_dict = self.tokenizer.encode_plus(text=Q1,
                                                    text_pair=Q2,
                                                    max_length=max_length,
                                                    pad_to_max_length=True)
        return np.array(tokenzied_dict['input_ids']), np.array(tokenzied_dict['token_type_ids']), np.array(
            tokenzied_dict['attention_mask']), np.array([data[2]])

    def __len__(self):
        return len(self.pairs)



