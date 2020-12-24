from transformers import BertTokenizer
from dataset.dataset import EvalDataBert
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def evaluateACC(model, test_file, tokenizer, params):
    eval_data = EvalDataBert(test_file=test_file,
                             s1_length=params['s1_length'],
                             s2_length=params['s2_length'],
                             max_length=params['max_length'],
                             tokenizer=tokenizer)

    eval_dataLoader = DataLoader(dataset=eval_data,
                                 batch_size=params['batch_size'],
                                 shuffle=False)
    model.eval()

    true_labels = []
    pred_labels = []


    for batch in tqdm(eval_dataLoader):

            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask, labels = batch

            with torch.no_grad():
                outputs = model(input_ids=input_ids.long(),
                                token_type_ids=token_type_ids.long(),
                                labels=labels)

                eval_loss, logits = outputs[:2]
                true_labels = true_labels + labels.cpu().numpy().tolist()
                for logit in logits:
                    if logit >= 0:
                        pred_labels += [[1]]
                    else:
                        pred_labels += [[0]]

    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)
    acc = ((pred_labels == true_labels).sum()) / len(pred_labels)

    return len(pred_labels), acc

if __name__ == '__main__':
    model_type = 'MatchModel'
    checkpoint_dir = '../results/result/model/checkpoint-3'
    test_file = '../../Olympic/data/test.txt'
    model_name = 'model.' + model_type

    BertMatchModel = __import__(model_name, globals(), locals(), model_type).BertMatchModel
    tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
    model = BertMatchModel.from_pretrained(checkpoint_dir)
    model.to(device)

    # 测试参数

    params = {'s1_length': 100,
              's2_length': 100,
              'max_length': None,
              'batch_size': 32}

    totol, acc = evaluateACC(model, test_file, tokenizer, params)

    print(f'在测试数据共{totol}条上，acc:{round(acc,2)}')