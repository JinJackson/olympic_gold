from elasticsearch5 import Elasticsearch
import hashlib
import json
from tqdm import tqdm
import time

es = Elasticsearch()

mapping = {
    "mappings": {
        "data": {
            "properties": { #属性
                "question": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                },
                "qtype": {
                    "type": "keyword"
                },
                "answer": {
                    "type": "text"
                },
                "timestamp": {
                    "type": "text"
                },
                "origin": {
                    "type": "text"
                },
            }
        }
    }
}

try:
    es.indices.delete(index='oly')
except:
    pass
finally:
    es.indices.create(index='oly',ignore=400,body=mapping)


# 保存文件函数
def saveFile(path, data):
    file = open(path, 'a') 
    file.write(json.dumps(data, ensure_ascii=False))
    file.write('\n')
    file.close()

data_path = "../data/oly.json"

def md5(question):
    encoder = hashlib.md5()
    encoder.update(question.encode('utf-8'))
    return encoder.hexdigest()

count=0
digit = 0
rm = 0
without_test = True

test_q_lis = []
# if without_test:
#     with open('/home/olympic/Data/test_set/test_v1.txt') as f:
#         for line in f:
#             test_q_lis.append(line.split()[0].strip())

f=open(data_path,"r",encoding="utf-8")

now_timestamp = time.time()
for line in tqdm(f.readlines()):

    line = json.loads(line)

    
    if "qtype" not in line['_source']:
        line['_source']['qtype'] = 'Which'

    data = [line['_source']['question'], line['_source']['answer'], line['_source']['qtype']]
    if len(data)==3:
        question=data[0]
        answer=data[1]
        qtype = data[2] if data[2] != "Where" else "Which"
        if question not in test_q_lis:
            body = {"question":question, "answer":answer, "qtype":qtype, "timestamp": str(now_timestamp), "origin": "baike"}
            md5Id = md5(question)
            if es.exists(index = 'oly', id = md5Id, doc_type = 'qapair'):
                rm += 1
                continue
            es.create(index='oly',body=body,id=md5Id,doc_type='qapair')
            count+=1
        else:
            rm += 1

print("add "+str(count)+" docs")
print("rm "+str(rm)+" docs")
f.close()
