import json
from torch.utils.data import Dataset
import codecs
import numpy as np


#加入了tag_embedding

def pad_sent(sent, pad, max_length):
    sent_arr = sent.split()
    sent_arr.extend([pad for i in range(max_length)])
    return ' '.join(sent_arr[:max_length]).strip()

def get_tagids(input_ids):
    sep_pos = input_ids.index(102)

    ids_part1 = input_ids[1:sep_pos]
    ids_part2 = input_ids[sep_pos + 1:-1]

    tag_ids1 = []
    tag_ids2 = []

    for ids in ids_part1:
        if ids in ids_part2:
            tag_ids1.append(1)
        else:
            tag_ids1.append(0)

    for ids in ids_part2:
        if ids in ids_part1:
            tag_ids2.append(1)
        else:
            tag_ids2.append(0)

    tag_ids = [1] + tag_ids1 + [1] + tag_ids2 + [1]

    return tag_ids


class TrainDataBert(Dataset):
    def __init__(self, train_file, s1_length, s2_length, max_length, tokenizer):
        self.s1_length = s1_length
        self.s2_length = s2_length
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.pairs = []

        with codecs.open(train_file, 'r', encoding='utf-8') as reader:
            lines = reader.readlines()
            for line in lines:
                if len(line) > 2:
                    a_data = line.split('  [SEP]  ')
                    text1 = a_data[0]
                    text2 = a_data[1]
                    label = int(a_data[2])
                    self.pairs.append((text1, text2, label))

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
        input_ids = tokenzied_dict['input_ids']
        tags_ids = get_tagids(input_ids)
        return np.array(tokenzied_dict['input_ids']), np.array(tokenzied_dict['token_type_ids']), np.array(
            tokenzied_dict['attention_mask']), np.array(tags_ids), np.array([data[2]])

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
            lines = reader.readlines()
            for line in lines:
                if len(line) > 2:
                    a_data = line.split('  [SEP]  ')
                    text1 = a_data[0]
                    text2 = a_data[1]
                    label = int(a_data[2])
                    self.pairs.append((text1, text2, label))

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
        input_ids = tokenzied_dict['input_ids']
        tags_ids = get_tagids(input_ids)
        return np.array(tokenzied_dict['input_ids']), np.array(tokenzied_dict['token_type_ids']), np.array(
            tokenzied_dict['attention_mask']), np.array(tags_ids), np.array([data[2]])

    def __len__(self):
        return len(self.pairs)
