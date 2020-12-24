## -*- coding: utf-8 -*-
from elasticsearch5 import Elasticsearch

class Searcher:
    def __init__(self):
        self.es = Elasticsearch()

    def searchQuery(self, question, qtype=None, index="oly"):
        if qtype != None:
            body = {'query': {'bool': {'must': [{'match': {'qtype': qtype}}, {'match': {'question': question}}]}}}
        else:
            body = {'query': {'match': {'question': question}}}
        res = self.es.search(index=index, body=body)['hits']["hits"]
        results = [question]
        for data in res:
            dic = {}
            dic['question'] = data["_source"]["question"]
            dic['answer'] = data["_source"]["answer"]
            if "timestamp" in data["_source"]:
                dic['timestamp'] = data["_source"]["timestamp"]
            else:
                dic['timestamp'] = "null"
            if "origin" in data["_source"]:
                dic['origin'] = data["_source"]["origin"]
            else:
                dic['origin'] = "base"
            # es中部分问题对没有qtype字段
            if "qtype" in data["_source"]:
                dic['qtype'] = data["_source"]["qtype"]
            else:
                dic['qtype'] = None
            dic['score'] = data["_score"]
            results.append(dic)
        return results

if __name__ == "__main__":
    searcher = Searcher()
    result = searcher.searchQuery("23届冬奥会1500米短道速滑冠军是谁")
    print(result[2])
    for data in result:
        print(data)