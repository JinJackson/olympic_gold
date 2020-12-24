#--do_train --bert_model  "bert-base-chinese" --model_type "MatchModelwithTag" --train_file "data/finetune/finetune2/train2.txt" --test_file "data/finetune/finetune2/test2.txt" --do_lower_case --learning_rate 1e-5 --gpu 0 --epochs 12 --batch_size 4 --s1_length 50 --s2_length 50 --seed 1024 --save_dir "result_Olympic2/model"
from parser1 import args

from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

import os, random
import glob
import torch

import numpy as np
from tqdm import tqdm

from dataset.dataset_Olym2 import TrainDataBert, EvalDataBert

from utils.logger import getLogger
#from utils.evaluateAcc import evaluateACC
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #if using multi-gpu
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn
    # torch.backends.cudnn.enabled = False

if args.seed > -1:
    seed_torch(args.seed)

model_name = 'model.' + args.model_type
BertMatchModel = __import__(model_name, globals(), locals(), [args.model_type]).BertMatchModel

logger = None

def train(model, tokenizer, checkpoint):
    #apex加速部分省略
    amp = None

    #训练数据处理
    train_data = TrainDataBert(train_file=args.train_file,
                               s1_length=args.s1_length,
                               s2_length=args.s2_length,
                               max_length = args.max_length,
                               tokenizer=tokenizer,
                               )
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  shuffle=not args.pair)

    # 初始化 opimizer, scheduler
    t_totol = len(train_dataloader) * args.epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_totol
    )

    # 读取断点 optimizer, scheduler
    checkpoint_dir = args.save_dir + '/checkpoint-' + str(checkpoint)
    if os.path.isfile(os.path.join(checkpoint_dir, 'optimizer.pt')):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'scheduler.pt')))

    # 开始训练
    logger.debug("***** Running training *****")
    logger.debug(" Num examples = %d", len(train_dataloader))
    logger.debug(" Num Epochs = %d", args.epochs)
    logger.debug(" Batch size = %d", args.batch_size)

    # 没有checkpoints就从0开始
    if checkpoint < 0:
        checkpoint = 0
    else:
        checkpoint += 1

    logger.debug(" Start Batch = %d", checkpoint)
    for epoch in range(checkpoint, args.epochs):
        model.train()
        epoch_loss = []

        for batch in tqdm(train_dataloader, desc='Iteration'):
            model.zero_grad()
            # 设置gpu运行
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, token_type_ids, attention_mask, tag_embeds, labels = batch

            outputs = model(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), tag_embeds=tag_embeds.long(), labels=labels)
            # print()
            # print('label', labels.data)
            # print('logits', outputs[1].data, 'loss', outputs[0].data)
            loss = outputs[0]

            epoch_loss.append(loss.item())

            loss.backward()

            optimizer.step()
            scheduler.step()

        # 保存模型
        output_dir = args.save_dir + "/checkpoint-" + str(epoch)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.debug('Saving model checkingpoint to %s', output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.debug('Saving optimizer and scheduler states to %s', output_dir)

        logger.info(
            f'【Train 】 Train Epoch {epoch}: train_loss={round(np.array(epoch_loss).mean(), 4)}')
        # eval dev
        test_loss, test_acc = test(model, tokenizer, test_file=args.test_file, checkpoint=epoch, output_dir=output_dir)
        # 输出日志+保存日志
        logger.info(
            f'【TEST】 Train Epoch {epoch}: train_loss={round(np.array(test_loss).mean(), 4)}, acc:{test_acc}')


def test(model, tokenizer, test_file, checkpoint, output_dir=None):
    #eval数据处理：eval可能是test或者dev?
    eval_data = EvalDataBert(test_file=test_file,
                             s1_length=args.s1_length,
                             s2_length=args.s2_length,
                             max_length=args.max_length,
                             tokenizer=tokenizer
                             )

    eval_dataLoader = DataLoader(dataset=eval_data,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    logger.debug("***** Running evaluation {} *****".format(checkpoint))
    logger.debug(" Num examples = %d", len(eval_dataLoader))
    logger.debug(" Batch size = %d", args.batch_size)

    loss = []


    all_labels = None
    all_logits = None
    model.eval()

    for batch in tqdm(eval_dataLoader, desc="testing"):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, token_type_ids, attention_mask, tag_embeds, labels = batch

        with torch.no_grad():
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            tag_embeds=tag_embeds.long(),
                            labels=labels)

            eval_loss, logits = outputs[:2]

            loss.append(eval_loss.item())

            if all_labels is None:
                all_labels = labels.detach().cpu().numpy()
                all_logits = logits.detach().cpu().numpy()

            else:
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
    all_predict = (all_logits > 0) + 0
    results = (all_predict == all_labels)
    acc = results.sum() / len(all_predict)


    return np.array(loss).mean(), acc

if __name__ == "__main__":
    #创建存储目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger = getLogger(__name__, os.path.join(args.save_dir, 'log.txt'))

    if args.do_train:
        #train  #接着未训练完成的checkpoint继续训练
        checkpoint = -1

        #这里是找到最大的checkpoint然后加载它
        for checkpoint_dir_name in glob.glob(args.save_dir + "/*"):
            try:
                checkpoint = max(checkpoint, int(checkpoint_dir_name.split('/')[-1].split('-')[1]))
            except Exception as e:
                pass

        checkpoint_dir = args.save_dir + '/checkpoint-' + str(checkpoint)
        if checkpoint > -1:
            logger.debug(f'Load Model from {checkpoint}')


        #看看有没有保存的tokenizer和model，没有的话就加载参数的
        tokenizer = BertTokenizer.from_pretrained(args.bert_model if checkpoint == -1 else checkpoint_dir)
        model = BertMatchModel.from_pretrained(args.bert_model if checkpoint == -1 else checkpoint_dir)
        model.to(args.device)
        #训练
        train(model, tokenizer, checkpoint)
        #评估
    else:
        checkpoint =args.checkpoint
        checkpoint_dir = args.save_dir + '/checkpoint-' + str(checkpoint)
        tokenizer = BertTokenizer.from_pretrained(checkpoint_dir, do_lower_case=args.do_lower_case)
        model = BertMatchModel.from_pretrained(checkpoint_dir)
        model.to(args.device)
        eval_loss, acc = test(model, tokenizer, test_file=args.test_file, checkpoint=checkpoint)
        logger.debug('Evaluate Epoch %d: loss=%.4f' % (checkpoint, eval_loss))
