import codecs
from torch.utils.data import Dataset
import numpy as np

# train_file = 'data/train.txt'
# test_file = 'data/test.txt'

def pad_sent(sent, pad, max_length):
    sent_arr = sent.split()
    sent_arr.extend([pad for i in range(max_length)])
    return ' '.join(sent_arr[:max_length]).strip()


class PairsData(Dataset):
    def __init__(self, QApairs, Input_Q, Q_length, tokenizer):
        self.Q_length = Q_length
        self.Input_Q = Input_Q
        self.max_length = 2 * Q_length + 3
        self.tokenizer = tokenizer
        self.pairs = QApairs


    def __getitem__(self, index):
        data = self.pairs[index]
        if self.max_length == 0:
            Q1 = pad_sent(self.Input_Q, '[PAD]', self.Q_length)
            Q2 = pad_sent(data[0], '[PAD]', self.Q_length)
            max_length = self.max_length
        else:
            Q1 = self.Input_Q
            Q2 = data[0]
            max_length = self.max_length

        tokenzied_dict = self.tokenizer.encode_plus(text=Q1,
                                                    text_pair=Q2,
                                                    truncation=True,
                                                    max_length=max_length,
                                                    padding='max_length')
        return np.array(tokenzied_dict['input_ids']), np.array(tokenzied_dict['token_type_ids']), np.array(
            tokenzied_dict['attention_mask']), data[1]

    def __len__(self):
        return len(self.pairs)


