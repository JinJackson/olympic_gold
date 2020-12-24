from transformers import BertTokenizer
import torch
from utils.search_index import Searcher
import numpy as np
import time
from utils.logger import getLogger
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = None


#model_name = 'model.MatchModel'  (模型所在包.模型文件)
#model_type = 'MatchModel'   (模型文件名）
def load_model(model_name, model_type, checkpoint_dir):
    BertMatchModel = __import__(model_name, globals(), locals(), model_type).BertMatchModel
    tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
    model = BertMatchModel.from_pretrained(checkpoint_dir)
    model.to(device)

    return model, tokenizer

def pad_sent(sent, pad, max_length):
    sent_arr = sent.split()
    sent_arr.extend([pad for i in range(max_length)])
    return ' '.join(sent_arr[:max_length]).strip()


def get_Top10(input_Q):
    QApairs = []
    searcher = Searcher()
    top10 = searcher.searchQuery(input_Q)
    for i in range(1, len(top10)):
        data = top10[i]
        Q = data['question']
        A = data['answer']
        QApairs.append([Q, A])

    return QApairs

def get_Best(model, tokenizer, Q_length, input_Q, QApairs):
    max_length = 2 * Q_length + 3
    Q1 = input_Q
    Q1 = pad_sent(Q1, '[PAD]', Q_length)

    max = float('-inf')   #记录最大的相似度
    ans = None  #记录最大相似度对应的问题的答案
    result = []

    for i in range(len(QApairs)):
        QApair = QApairs[i]
        Q2 = QApair[0]
        Q2 = pad_sent(Q2, '[PAD]', Q_length)
        tokenzied_dict = tokenizer.encode_plus(text=Q1,
                                                text_pair=Q2,
                                                max_length=max_length,
                                                pad_to_max_length=True)
        input_ids, token_type_ids, attention_mask = torch.LongTensor(tokenzied_dict['input_ids']).unsqueeze(0).to(device), \
                                                    torch.LongTensor(tokenzied_dict['token_type_ids']).unsqueeze(0).to(device), \
                                                    torch.Tensor(tokenzied_dict['attention_mask']).unsqueeze(0).to(device)

        _, similarity = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        result.append((QApair[0], QApair[1], round(similarity.item(),4)))
        if similarity >= max:
            max = similarity
            ans = QApair[1]
    result = sorted(result, key=lambda x:x[2], reverse=True)
    return max, ans, result


if __name__ == '__main__':
    #print(time.asctime(time.localtime(time.time())))
    logger = getLogger(__name__, os.path.join('log', 'run_log.txt'))

    #载入模型
    time1 = time.time()
    #print('loading model...')
    #timestamp = str(time.asctime(time.localtime(time.time())))
    logger.info(f'loading model... ')
    model_name = 'model.MatchModel'
    model_type = 'MatchModel'
    checkpoint_dir = 'results/result/model/checkpoint-5'
    model, tokenizer = load_model(model_name=model_name, model_type=model_type, checkpoint_dir=checkpoint_dir)
    Q_length = 100
    #print(f'loading time = {loading_time}')
    loading_time = round(time.time()-time1,4)
    timestamp = str(time.asctime(time.localtime(time.time())))
    logger.info(f'timecost = {loading_time}')



    #输入问题，得到答案
    #input_Q = '23届冬奥会短道速滑1500冠军是谁？'
    while True:
        input_Q = input('请输入问题：')
        #timestamp = str(time.asctime(time.localtime(time.time())))
        logger.info(f'Input Question:{input_Q}')
        #print('getting Answer...')
        time2 = time.time()
        QApairs = get_Top10(input_Q)
        max_sim, ans, results = get_Best(model=model, tokenizer=tokenizer, Q_length=Q_length, input_Q=input_Q, QApairs=QApairs)
        search_time = round(time.time()-time2 ,4)

        timestamp = str(time.asctime(time.localtime(time.time())))
        logger.info(f'【Final Answer】:{ans}, searchtime:{search_time}')
        #print(results)
        print(ans)
        logger.info('【Candidates】Candidates Answers:')
        for i, result in enumerate(results):
            logger.info(f'【Cand{i+1}】：Q：{result[0]}, A:{result[1]}, Similarity:{result[2]}')
        # print(f'Search time:{search_time}')
        logger.info('-----------------------------------------------------------------------------------------------------------------')



