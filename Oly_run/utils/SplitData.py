import random
import codecs


data_file = '../../Olympic/data/finetune/finetune2/QApairs.txt'
train_file = '../../Olympic/data/finetune/finetune2/train.txt'
test_file = '../../Olympic/data/finetune/finetune2/test.txt'

datas = []
with codecs.open(data_file, 'r', encoding='utf-8') as reader:
    lines = reader.readlines()
    for line in lines:
        if len(line) > 2:
          a_data = line.split('  [SEP]  ')
          datas.append((a_data[0], a_data[1], int(a_data[2])))


random.shuffle(datas)

counts = len(datas)

train_data = datas[:int(0.8*counts)]
test_data = datas[int(0.8*counts):]

with codecs.open(train_file, 'w', encoding='utf-8') as writer:
    for data in train_data:
        text1 = data[0]
        text2 = data[1]
        label = str(data[2])
        writer.write(text1 + '  [SEP]  '+ text2 + '  [SEP]  '+ label + '\n')

with codecs.open(test_file, 'w', encoding='utf-8') as writer:
    for data in test_data:
        text1 = data[0]
        text2 = data[1]
        label = str(data[2])
        writer.write(text1 + '  [SEP]  '+ text2 + '  [SEP]  '+ label + '\n')