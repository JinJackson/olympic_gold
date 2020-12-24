import json
from utils.search_index import Searcher

searcher = Searcher()

def getAllDatas(data_path):
    datasets = []
    with open('../data/all_result.json', 'r', encoding='utf8') as reader:
        content = reader.read()[2:-2]
        datas = content.split('}, {')

        for data in datas:
            data = '{' + data + '}'
            data = json.loads(data)
            datasets.append(data)

    return datasets

    #dataset里的数据格式
    # {'pre_handle': 'saved', 'answer': '  夏洛特·卡拉', 'question': '瑞典代表团的谁在第23届冬奥会获得越野滑雪女子15公里追逐赛金牌?', 'id': 0,
    #  'qtype': 'Who', 'origin_id': 0, 'itype': '事实'}

#searcher = Searcher()

def getDataset(all_datas):
    results = {}
    for data in all_datas:
        Q = data['question']
        A = data['answer']
        Q_id = data['origin_id']
        if Q_id not in results.keys():
            results[Q_id] = []
            results[Q_id].append(A)
            results[Q_id].append(Q)
        else:
            results[Q_id].append(Q)

    print(len(results))

    pops = []
    for item in results.items():
        if len(item[1]) <= 2:
            pops.append(item[0])
    for pop in pops:
        results.pop(pop)

    print(len(results))
    #print(results)
    return results


def getQApairs(dataset):
    QApairs = []
    for items in dataset.items():
        datas = items[1]
        A = datas[0].lstrip().rstrip()
        for i in range(1,len(datas)-1):
            for j in range(i+1, len(datas)):
                Q1 = datas[i]
                Q2 = datas[j]
                QApairs.append((Q1, Q2, A, 1))
        Q_base = datas[1]
        top10 = searcher.searchQuery(Q_base)[1:]
        # print(Q_base)
        # print(A)
        for record in top10:
            #print(record['answer'])
            # label = None
            sim_Q = record['question']
            sim_A = record['answer'].rstrip().lstrip()
            if sim_Q == Q_base:
                continue
            else:
                if sim_A != A:
                    QApairs.append((Q_base, sim_Q, 0))
        QApairs.append('c')

        #print(Q_base)
    return QApairs


data_path = '../data/all_result.json'
all_datas = getAllDatas(data_path)
dataset = getDataset(all_datas)
QApairs = getQApairs(dataset)

count = 0
pos = 0
neg = 0
with open('../data/QApairs.txt', 'w', encoding='utf-8') as writer:
    for QApair in QApairs:
        if len(QApair) == 1:
            writer.write('\n')
        else:
            Q1 = QApair[0]
            Q2 = QApair[1]
            if len(QApair) == 4:
                label = QApair[3]
            else:
                label = QApair[2]
            if label == 0:
                neg += 1
            else:
                pos += 1
            writer.write(Q1 + '  [SEP]  ' + Q2 + '  [SEP]  '+ str(label) + '\n')
            count += 1
print(count)
print(pos, neg)

#print(QApairs)
#print(len(QApairs))
#print(results)
    # top10 = searcher.searchQuery(Q)
    # for candidate in top10[1:]:
    #     print(candidate)
    #     break
#print(QApairs)
