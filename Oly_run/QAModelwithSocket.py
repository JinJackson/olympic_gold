## -*- coding: utf-8 -*-
from transformers import BertTokenizer
import torch
from utils.search_index import Searcher
import time
from utils.logger import getLogger
import os
import json
from dataset.datasetForSocket import PairsData
#from dataset.datasetForSocket4trans211 import PairsData
from torch.utils.data import DataLoader
import socket
from sys import argv
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = None
searcher = Searcher()

#model_name = 'model.MatchModel'  (模型所在包.模型文件)
#model_type = 'MatchModel'   (模型文件名）

port = int(argv[1])
print(port)
def load_model(model_name, model_type, checkpoint_dir):
    BertMatchModel = __import__(model_name, globals(), locals(), model_type).BertMatchModel
    tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
    model = BertMatchModel.from_pretrained(checkpoint_dir)
    #model.to(device)
    #model.to('cuda')

    return model, tokenizer

def pad_sent(sent, pad, max_length):
    sent_arr = sent.split()
    sent_arr.extend([pad for i in range(max_length)])
    return ' '.join(sent_arr[:max_length]).strip()


def get_Top10(input_Q):   #返回Top10问题答案对子
    QApairs = []

    top10 = searcher.searchQuery(input_Q)
    for i in range(1, len(top10)):
        data = top10[i]
        Q = data['question']
        A = data['answer']
        QApairs.append([Q, A])

    return QApairs


def get_Best(model, tokenizer, Q_length, input_Q, QApairs):
    pairDataset = PairsData(QApairs, input_Q, Q_length, tokenizer)
    dataLoader = DataLoader(dataset=pairDataset,
                            batch_size=10,
                            shuffle=False)
    answers = None
    scores = None
    for batch in dataLoader:
        inputs = batch[:3]
        #inputs = tuple(t.to('cuda') for t in inputs)
        input_ids, token_type_ids, attention_mask = inputs

        _, scores = model(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask)
        answers = batch[-1]
        scores = scores.detach().cpu().numpy().tolist()
    for QApair, score in zip(QApairs, scores):
        QApair += score
    QApairs.sort(key=lambda x:x[2], reverse=True)

    question, ans, max_sim = QApairs[0]
    return max_sim, question, ans, QApairs



def pack_data(max, question, ans, result):
    send_data = {}
    send_data['question'] = question
    send_data['answer'] = ans
    send_data ['answerID'] = 1
    candidate_lis = []
    for i , QApair in enumerate(result):
        candidate_dic = {}
        candidate_dic['question'] = QApair[0]
        candidate_dic['answer'] = QApair[1]
        candidate_dic["timestamp"] = '0'
        candidate_dic["origin"] = '0'
        candidate_dic['whooshRank'] = int(QApair[2])
        candidate_dic['rank'] = int(QApair[2])
        candidate_dic['score'] = int(QApair[2])
        candidate_lis.append(candidate_dic)
    send_data['candidates'] = candidate_lis
    s = json.dumps(send_data)
    return s

if __name__ == '__main__':
    #print(time.asctime(time.localtime(time.time())))
    logger = getLogger(__name__, os.path.join('../Oly_run/log', 'run_log.txt'))

    #载入模型
    time1 = time.time()
    #print('loading model...')
    #timestamp = str(time.asctime(time.localtime(time.time())))
    logger.info(f'loading model... ')
    model_name = 'model.MatchModel'
    model_type = 'MatchModel'
    checkpoint_dir = 'checkpoints/checkpoint-5'
    model, tokenizer = load_model(model_name=model_name, model_type=model_type, checkpoint_dir=checkpoint_dir)
    Q_length = 100
    #print(f'loading time = {loading_time}')
    loading_time = round(time.time()-time1,4)
    timestamp = str(time.asctime(time.localtime(time.time())))
    logger.info(f'timecost = {loading_time}')


    # #时间测试
    input_Q = '23届冬奥会1500米短道速滑冠军是谁？'

    start_time = time.time()
    QApairs = get_Top10(input_Q)
    print('estime', time.time()-start_time)

    start_time = time.time()
    max_sim, question, ans, results = get_Best(model=model, tokenizer=tokenizer, Q_length=Q_length, input_Q=input_Q,
                                               QApairs=QApairs)
    print('modeltime', time.time()-start_time)
    print(max_sim, question, ans)
    print(results)


    # # 在构建socket的时候需要用到ip和端口，必须是元组的形式。
    # # 另外，因为是本机上的两个程序通信，所以设置成localhost，
    # # 如果要和网络上的其他主机进行通信，则填上相应主机的ip地
    # # 址，端口的话随便设置一个，不要和已知的一些服务冲突就行
    # address = ('localhost', port)
    # # 创建socket对象，同时设置通信模式，AF_INET代表IPv4，SOCK_STREAM代表流式socket，使用的是tcp协议
    # server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # # 绑定到我们刚刚设置的ip和端口元组，代表我们的服务运行在本机的9999端口上
    # server.bind(address)
    #
    # # 开始监听，5位最大挂起的连接数
    # server.listen(5)
    #
    #
    #
    #
    # #输入问题，得到答案
    # #input_Q = '23届冬奥会短道速滑1500冠军是谁？'
    # print("服务启动")
    # while True:
    #     print("server waiting")
    #     # accept()方法被动接受客户端连接，阻塞，等待连接. client是客户端的socket对象，可以实现消息的接收和发送，addr表示客户端的地址
    #     client, addr = server.accept()
    #     data = client.recv(1024)  # 代表从发过来的数据中读取13byte的数据
    #     input_Q = str(data, encoding='utf-8')
    #     print(input_Q)
    #     #timestamp = str(time.asctime(time.localtime(time.time())))
    #     logger.info(f'Input Question:{input_Q}')
    #     #print('getting Answer...')
    #     time2 = time.time()
    #     QApairs = get_Top10(input_Q)
    #
    #     max_sim, question, ans, results = get_Best(model=model, tokenizer=tokenizer, Q_length=Q_length, input_Q=input_Q, QApairs=QApairs)
    #
    #     search_time = round(time.time()-time2, 4)
    #     send_data = pack_data(max_sim, question, ans, results)
    #     timestamp = str(time.asctime(time.localtime(time.time())))
    #
    #     logger.info(f'【Final Answer】:{ans}, searchtime:{search_time}')
    #     #print(results)
    #     client.sendall(bytes(send_data, encoding='gbk'))  # 发送消息给客户端，发送的消息必须是byte类型
    #     client.close()  # 关闭连接
    #     logger.info('【Candidates】Candidates Answers:')
    #     for i, result in enumerate(results):
    #         logger.info(f'【Cand{i+1}】：Q：{result[0]}, A:{result[1]}, Similarity:{result[2]}')
    #     # print(f'Search time:{search_time}')
    #     logger.info('-----------------------------------------------------------------------------------------------------------------')
    #
    #

