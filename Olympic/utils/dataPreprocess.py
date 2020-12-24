import json
from utils.search_index import Searcher
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

random.seed(1024)
datapath  ="data/oly-all-19-11-26.json"

searcher = Searcher()
def readQApairs(datapath):
    QApairsData = []
    with open(datapath, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
        # {"_index": "oly", "_type": "qapair", "_id": "bb76334c6abb6c14c973d227395d6177", "_score": 1,
        #  "_source": {"question": "第23届冬奥会短道速滑男子1500米比赛的前一届举办时间是什么时候？", "answer": "2014年", "qtype": "When",
        #              "timestamp": "1573694900.6121526", "origin": "base"}}
        for line in lines:
            line = json.loads(line)
            Q = line['_source']['question']
            A = line['_source']['answer']
            if Q and A:
                # print(line['_source'])
                QApairsData.append((Q, A))

    return  QApairsData


def generateData(QApairsData):
    dataset = []
    pos = 0
    neg = 0
    for QApair in tqdm(QApairsData):
        if len(QApair) != 2:
            continue
        Base_Q = QApair[0]
        Base_A = QApair[1]
        top10 = searcher.searchQuery(Base_Q)  #len=11,其中第0个位置是BASE_Q， 剩下十条的数据形式如下：
        #{'question': '谁是2014年22届冬奥会短道速滑1500米冠军？', 'answer': '周洋', 'timestamp': '1602684112.2720432', 'origin': 'baike', 'qtype': 'Who', 'score': 25.670996}
        for i in range(1, len(top10) - 1):
            data = top10[i]
            sim_Q = data['question']
            sim_A = data['answer']
            if sim_Q and sim_A:
                label = 1 if Base_A == sim_A else 0
                if Base_Q >= sim_Q:
                    sample = (Base_Q, sim_Q, label)
                else:
                    sample = (sim_Q, Base_Q, label)
                dataset.append(sample)

                if label == 0:
                    neg += 1
                else:
                    pos +=1

    print()
    print('len_Dataset', str(len(dataset)), ',pos:', str(pos), 'neg:', str(neg))
    pos = 0
    neg = 0
    dataset = list(set(dataset))
    for data in dataset:
        if data[2] == 0:
            neg += 1
        else:
            pos += 1
    print('去重后：')
    print('len_Dataset', str(len(dataset)), ',pos:', str(pos), 'neg:', str(neg))


    with open('../data/all_data.txt', 'w', encoding='utf-8') as writer:
        print('writing')
        for data in tqdm(dataset):
            line = str(data[0]) + '  [SEP]  '+ str(data[1]) + '  [SEP]  ' + str(data[2]) + '\n'
            writer.write(line)

def splitDataset():
    dataset = []
    with open('../data/all_data.txt', 'r', encoding='utf-8') as reader:
        all_data = reader.readlines()
        for line in all_data:
            if line:
                dataset.append(line)
        random.shuffle(all_data)
        train_data_size = int(0.8 * len(dataset))

        train_data = all_data[:train_data_size]
        test_data = all_data[train_data_size:]
        print(len(train_data), len(test_data))
    with open('../data/train.txt', 'w', encoding='utf-8') as writer:
        writer.writelines(train_data)

    with open('../data/test.txt', 'w', encoding='utf-8') as writer:
        writer.writelines(test_data)





if __name__ == '__main__':
    QApairsData = readQApairs(datapath)
    generateData(QApairsData)
    splitDataset()