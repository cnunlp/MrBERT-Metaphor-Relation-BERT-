# -*- coding: utf-8 -*-
import os
import csv
import ast
import torch
import random
import logging
from numpy import *
import numpy as np
import torch.nn.functional as F
import argparse
from tokenizers import BertWordPieceTokenizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from cosine_model import MrBERT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging = logging.getLogger(__name__)

"""
1. Data pre-processing
"""
def load_vua():
    """ 读取 VUA 数据
    """
    with open('./data/VUA/VUA_train_noVAL_labels.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        train_svo_labels, train_seq_labels = [], []
        for line in lines:
            train_seq_labels.append(ast.literal_eval(line[0]))
            train_svo_labels.append(ast.literal_eval(line[1]))
    with open('./data/VUA/VUA_val_labels.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        val_svo_labels, val_seq_labels = [], []
        for line in lines:
            val_seq_labels.append(ast.literal_eval(line[0]))
            val_svo_labels.append(ast.literal_eval(line[1]))
    with open('./data/embeddings_vua/vua_train_embeddings_ave.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        train_embeddings = []
        for line in lines:
            train_embeddings.append(ast.literal_eval(line[1]))
    with open('./data/embeddings_vua/vua_val_embeddings_ave.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        val_embeddings = []
        for line in lines:
            val_embeddings.append(ast.literal_eval(line[1]))

    raw_vua_train=[]
    raw_vua_val=[]

    with open('./data/VUA/VUA_formatted_train_noVAL_final.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        i = 0
        for line in lines:
            sen = line[3].split()
            v_pos = int(line[4])
            s_pos = -1
            o_pos = -1
            if 1 in train_svo_labels[i]:
                s_pos = train_svo_labels[i].index(1)
            if 3 in train_svo_labels[i]:
                o_pos = train_svo_labels[i].index(3)
            label_seq = train_seq_labels[i]
            assert len(label_seq) == len(sen)
            assert len(train_svo_labels[i]) == len(sen)
            raw_vua_train.append([line[3].strip(), label_seq, s_pos, v_pos, o_pos , sen[v_pos], int(line[5]), train_embeddings[i]])
            i += 1

    with open('./data/VUA/VUA_formatted_val_final.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        i = 0
        for line in lines:
            sen = line[3].split()
            v_pos = int(line[4])
            s_pos = -1
            o_pos = -1
            if 1 in val_svo_labels[i]:
                s_pos = val_svo_labels[i].index(1)
            if 3 in val_svo_labels[i]:
                o_pos = val_svo_labels[i].index(3)
            label_seq = val_seq_labels[i]
            assert (len(label_seq) == len(sen))
            assert len(val_svo_labels[i]) == len(sen)
            raw_vua_val.append([line[3].strip(), label_seq, s_pos, v_pos, o_pos, sen[v_pos], int(line[5]), val_embeddings[i]])
            i += 1

    return raw_vua_train, raw_vua_val

def load_vua_test():
    """ 读取 VUA 测试集数据
    """
    test_svo_labels ,test_seq_labels = [], []
    with open('./data/VUA/VUA_test_labels.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            test_seq_labels.append(ast.literal_eval(line[0]))
            test_svo_labels.append(ast.literal_eval(line[1]))
    with open('./data/embeddings_vua/vua_test_embeddings_ave.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        test_embeddings = []
        for line in lines:
            test_embeddings.append(ast.literal_eval(line[1]))
    with open('./data/VUA/VUA_formatted_test_final.csv', encoding='utf8') as f:
        raw_vua_test = []
        lines = csv.reader(f)
        next(lines)
        i = 0
        for line in lines:
            sen = line[3].split()
            v_pos = int(line[4])
            s_pos = -1
            o_pos = -1
            if 1 in test_svo_labels[i]:
                s_pos = test_svo_labels[i].index(1)
            if 3 in test_svo_labels[i]:
                o_pos = test_svo_labels[i].index(3)
            label_seq = test_seq_labels[i]
            assert (len(label_seq) == len(sen))
            assert len(test_svo_labels[i]) == len(sen)
            raw_vua_test.append([line[3].strip(), label_seq, s_pos, v_pos, o_pos, int(line[5]),line[0]+'_'+line[1], test_embeddings[i]])
            i += 1
    return raw_vua_test

def insert_tag(sentences, s_pos, v_pos, o_pos):
    '''
    准确插入！可能有多个subj/verb/obj
    :param sentences:
    :param s_pos:
    :param v_pos:
    :param o_pos:
    :return:
    '''
    tokenized_texts=[]
    for i in range(len(sentences)):
        sen = sentences[i].split()
        v = sen[v_pos[i]]
        sen[v_pos[i]] = '[verb] '+ v + ' [/verb]'
        if not (s_pos[i] == -1) and not (o_pos[i] == -1):
            s = sen[s_pos[i]]
            sen[s_pos[i]] = '[subj] ' + s + ' [/subj]'
            o = sen[o_pos[i]]
            sen[o_pos[i]] = '[obj] ' + o + ' [/obj]'
        elif not s_pos[i]==-1 and o_pos[i]==-1:
            s = sen[s_pos[i]]
            sen[s_pos[i]] = '[subj] ' + s + ' [/subj]'
        elif s_pos[i]==-1 and not o_pos[i]==-1:
            o = sen[o_pos[i]]
            sen[o_pos[i]] = '[obj] ' + o + ' [/obj]'
        txt = (' '.join(sen)).split()
        tokenized_texts.append(txt)
    return tokenized_texts

def get_inputs(tokenizer, texts, labels0, labels2, max_len, embeddings):
    '''
    对输入进行 encode 和长度固定（截长补短）
    '''
    ids, labels = [], []
    i=0
    for txt in texts:
        id=[101]
        label=[-100]
        j=0
        for w in txt:
            enc = tokenizer.encode(w)
            id_w = enc.ids
            id.extend(id_w[1:len(id_w) - 1])
            if w =='[subj]' or w == '[/subj]' or w == '[verb]' or w == '[/verb]' or w =='[obj]' or w == '[/obj]':
                l=[-100]
            else:
                l = [labels0[i][j]]
                if len(enc.tokens)>3:
                    for t in range(2,len(id_w)-1):
                        l.append(-100)
                j += 1
            label.extend(l)
        id.append(102)
        label.append(-100)
        assert len(labels0[i]) == (len(label) - label.count(-100))
        assert len(label)==len(id)
        id = id + [0] * (max_len-len(id))
        label = label + [-100] * (max_len-len(label))
        ids.append(id)
        labels.append(label)
        i+=1
    input_ids = torch.tensor([[i for i in id] for id in ids])
    labels = torch.tensor([[i for i in label] for label in labels])
    labels2 = torch.tensor(labels2, dtype=torch.float)
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    # ! 设置 mask_attention
    masks = torch.tensor([[float(i > 0) for i in input_id]
                             for input_id in input_ids])
    return input_ids, labels, labels2, masks, embeddings

def evaluation(labels2,preds2):

    accuracy2 = accuracy_score(labels2, preds2)
    precision2 = precision_score(labels2, preds2)
    recall2 = recall_score(labels2, preds2)
    f12 = f1_score(labels2, preds2)

    print("{:15}{:<.3f}".format('accuracy:', accuracy2))
    print("{:15}{:<.3f}".format('precision:', precision2))
    print("{:15}{:<.3f}".format('recall:', recall2))
    print("{:15}{:<.3f}".format('f1', f12))

    return accuracy2, precision2, recall2, f12

def train(tr_raw_vua,val_raw_vua, test_raw_vua, device, model_dir, epochs, max_len, batch_size, learning_rate, max_grad_norm, repr, relmodel, dropout_rate):
    '''模型训练
    '''
    tr_sentences = [r[0] for r in tr_raw_vua]
    val_sentences = [r[0] for r in val_raw_vua]

    tr_labels0 = [r[1] for r in tr_raw_vua]
    val_labels0 = [r[1] for r in val_raw_vua]

    tr_s_pos = [r[2] for r in tr_raw_vua]
    val_s_pos = [r[2] for r in val_raw_vua]

    tr_v_pos = [r[3] for r in tr_raw_vua]
    val_v_pos = [r[3] for r in val_raw_vua]

    tr_o_pos = [r[4] for r in tr_raw_vua]
    val_o_pos = [r[4] for r in val_raw_vua]

    tr_labels2 = [[r[6]] for r in tr_raw_vua]
    val_labels2 = [[r[6]] for r in val_raw_vua]

    tr_embeddings = [[r[7]] for r in tr_raw_vua]
    val_embeddings = [[r[7]] for r in val_raw_vua]

    test_sentences = [r[0] for r in test_raw_vua]

    test_labels0 = [r[1] for r in test_raw_vua]

    test_s_pos = [r[2] for r in test_raw_vua]

    test_v_pos = [r[3] for r in test_raw_vua]

    test_o_pos = [r[4] for r in test_raw_vua]

    test_labels2 = [[r[5]] for r in test_raw_vua]

    test_embeddings = [[r[7]] for r in test_raw_vua]

    tr_tokenized_texts = insert_tag(tr_sentences, tr_s_pos, tr_v_pos, tr_o_pos)

    val_tokenized_texts = insert_tag(val_sentences, val_s_pos, val_v_pos, val_o_pos)

    test_tokenized_texts = insert_tag(test_sentences, test_s_pos, test_v_pos, test_o_pos)

    tokenizer = BertWordPieceTokenizer('./vocab.txt')
    tokenizer.add_special_tokens(['[subj]','[/subj]', '[verb]','[/verb]','[obj]','[/obj]'])
    config = BertConfig.from_pretrained(model_dir + '/config.json')
    config.num_labels1 = 2
    config.num_labels2 = 2
    config.hidden_dropout_prob = dropout_rate

    print('----------config---------')
    print(config)
    model = MrBERT.from_pretrained(model_dir, config=config, bert_model_dir=model_dir, device=device, repr=repr, relmodel=relmodel)

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    # ! get inputs
    tr_input_ids, tr_labels1, tr_labels2, tr_masks, tr_embeddings = get_inputs(tokenizer, tr_tokenized_texts, tr_labels0, tr_labels2, max_len, tr_embeddings)
    val_input_ids, val_labels1, val_labels2, val_masks, val_embeddings = get_inputs(tokenizer, val_tokenized_texts, val_labels0, val_labels2, max_len, val_embeddings)
    test_input_ids, test_labels1,test_labels2, test_masks, test_embeddings = get_inputs(tokenizer, test_tokenized_texts, test_labels0, test_labels2, max_len, test_embeddings)

    train_data = TensorDataset(tr_input_ids, tr_masks, tr_labels1, tr_labels2, tr_embeddings)
    train_loader = DataLoader(train_data,  batch_size=batch_size,shuffle=True)

    val_data = TensorDataset(val_input_ids, val_masks, val_labels1, val_labels2, val_embeddings)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    test_data = TensorDataset(test_input_ids, test_masks, test_labels1, test_labels2, test_embeddings)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    # ! 定义 optimizer
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate)

    t_total = len(tr_input_ids) / batch_size * epochs + 1
    num_warmup_steps = int(t_total / 10) * 2
    logging.info('t_total: %d warmup: %d' % (t_total, num_warmup_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    val_accs2, val_ps2, val_rs2, val_f1s2, val_results1,val_results2  = [],[],[],[],[],[]
    eval_accs2, eval_ps2, eval_rs2, eval_f1s2, test_results1, test_results2 = [], [], [], [], [], []
    val_max_f1 = 0

    for epoch in range(epochs):
        print('===== Start training: epoch {} ====='.format(epoch + 1))

        model.train()

        tr_loss,tr_loss1,tr_loss2 = 0,0,0
        nb_tr_steps = 0

        # ! training
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels1, b_labels2, b_embeddings = batch
            outputs1, outputs2 = model(input_ids=b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask, labels1=b_labels1, labels2=b_labels2, embeddings=b_embeddings)
            loss = outputs1[0] + outputs2[0]

            loss.backward()

            tr_loss += float(loss.item())
            tr_loss1 += float(outputs1[0].item())
            tr_loss2 += float(outputs2[0].item())

            nb_tr_steps += 1

            # ! 减小梯度 https://www.cnblogs.com/lindaxin/p/7998196.html
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # ! 更新参数
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        print("\nEpoch {} of training loss: {}".format(epoch + 1, tr_loss / nb_tr_steps))
        print("\nEpoch {} of training loss1: {}".format(epoch + 1, tr_loss1 / nb_tr_steps))
        print("\nEpoch {} of training loss2: {}".format(epoch + 1, tr_loss2 / nb_tr_steps))

        # ! Validation
        model.eval()
        val_accuracy2, val_precision2, val_recall2, val_f12 = 0, 0, 0, 0
        nb_val_steps = 0

        val_preds1, val_labels1, val_preds2, val_labels2 = [], [], [], []
        for step, batch in enumerate(val_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels1, b_labels2, b_embeddings = batch
            with torch.no_grad():
                outputs1, outputs2 = model(input_ids=b_input_ids, token_type_ids=None,
                                           attention_mask=b_input_mask, labels1=b_labels1, labels2=b_labels2, embeddings=b_embeddings)

                tmp_val_loss1, logits1 = outputs1[:2]
                tmp_val_loss2, logits2 = outputs2[:2]

            # ! detach的方法，将variable参数从网络中隔离开，不参与参数更新
            logits1 = logits1.detach().cpu().numpy()
            logits2 = logits2.detach().cpu().numpy()
            ture_labels1 = b_labels1.cpu().numpy()
            ture_labels2 = b_labels2.cpu().numpy()

            # values1, logits1 = torch.max(F.softmax(logits1, dim=-1), dim=-1)[:2]

            for i in range(len(logits2)):
                pred2 = logits2[i][0]
                if pred2>0.5:
                    pred2=torch.Tensor([1])
                else:
                    pred2=torch.Tensor([0])
                val_preds2.append(int(pred2))
                val_labels2.append(int(ture_labels2[i]))
            nb_val_steps += 1

            # val_loss += tmp_val_loss.mean().item()

        # ! 计算评估值
        val_preds2 = np.array(val_preds2)
        val_labels2 = np.array(val_labels2)

        # 打印信息
        val_accuracy2, val_precision2, val_recall2, val_f12 = evaluation(val_labels2, val_preds2)

        val_accs2.append(val_accuracy2)
        val_ps2.append(val_precision2)
        val_rs2.append(val_recall2)
        val_f1s2.append(val_f12)

        # ! Test
        eval_loss, eval_accuracy1, eval_precision1, eval_recall1, eval_f11, eval_accuracy2, eval_precision2, eval_recall2, eval_f12 = 0, 0, 0, 0, 0, 0, 0, 0, 0
        preds1, labels1, preds2, labels2, t_labels1, p_labels1,t_labels2, p_labels2 = [], [], [],[] , [], [], [], []
        for step, batch in enumerate(test_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels1, b_labels2, b_embeddings = batch

            with torch.no_grad():
                outputs1, outputs2 = model(input_ids=b_input_ids, token_type_ids=None,
                                           attention_mask=b_input_mask, labels1=b_labels1, labels2=b_labels2, embeddings=b_embeddings)
                tmp_eval_loss1, logits1 = outputs1[:2]
                tmp_eval_loss2, logits2 = outputs2[:2]

            values1, logits1 = torch.max(F.softmax(logits1, dim=-1), dim=-1)[:2]

            # ! detach的方法，将variable参数从网络中隔离开，不参与参数更新
            logits1 = logits1.detach().cpu().numpy()
            logits2 = logits2.detach().cpu().numpy()
            ture_labels1 = b_labels1.cpu().numpy()
            ture_labels2 = b_labels2.cpu().numpy()

            for i in range(len(logits2)):
                pred2 = logits2[i][0]
                pred1 = logits1[i]
                label1 = ture_labels1[i]
                if pred2>0.5:
                    pred2=torch.Tensor([1])
                else:
                    pred2=torch.Tensor([0])
                preds2.append(int(pred2))
                labels2.append(int(ture_labels2[i]))
                pre, label = [], []
                for j in range(len(label1)):
                    if not label1[j] == -100:
                        label.append(int(label1[j]))
                        pre.append(int(pred1[j]))
        # 打印信息
        print("--- Test ---")
        eval_accuracy2, eval_precision2, eval_recall2, eval_f12 = evaluation(labels2, preds2)

        eval_accs2.append(eval_accuracy2)
        eval_ps2.append(eval_precision2)
        eval_rs2.append(eval_recall2)
        eval_f1s2.append(eval_f12)

    print("===== Train Finished =====\n")

    index=val_f1s2.index(max(val_f1s2))
    print("{:15}{:<}".format("val max epoch", index+1))
    print("{:15}{:<.3f}".format("accuracy", val_accs2[index]))
    print("{:15}{:<.3f}".format("precision", val_ps2[index]))
    print("{:15}{:<.3f}".format("recall", val_rs2[index]))
    print("{:15}{:<.3f}".format("f1", val_f1s2[index]))

    index=eval_f1s2.index(max(eval_f1s2))
    print("{:15}{:<}".format("test max epoch", index+1))
    print("{:15}{:<.3f}".format("accuracy", eval_accs2[index]))
    print("{:15}{:<.3f}".format("precision", eval_ps2[index]))
    print("{:15}{:<.3f}".format("recall", eval_rs2[index]))
    print("{:15}{:<.3f}".format("f1", eval_f1s2[index]))

def main():
    """
    ? 2. 设置基本参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str,
                        required=False, help='选择设备')
    parser.add_argument('--seed', default=4, type=int,
                        required=False, help='输入种子数')
    parser.add_argument('--bert_base_model_dir', type=str, required=True, help='BERT模型目录')
    parser.add_argument('--max_len', default=150, type=int, required=False, help='句子最大长度')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='训练batch_size')
    parser.add_argument('--lr', default=5e-5, type=float, required=False, help='学习率')
    parser.add_argument('--dropout_rate', default=0.1, type=float, required=False)
    parser.add_argument('--num_epochs', default=10, type=int, required=False, help='训练epoch')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--repr', default='average', type=str, required=False, choices=['tag', 'average'], help='获取表示形式：tag/average')
    parser.add_argument('--relmodel', default='linear', type=str, required=False, choices=['linear', 'bilinear', 'nt'], help='模型：linear/bilinear/nt（neural tensor）')
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    # ? 种子数设置
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ! 用以保证实验的可重复性，使每次运行的结果完全一致
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_vua_train, raw_vua_val = load_vua()
    raw_vua_test = load_vua_test()
    train( raw_vua_train, raw_vua_val, raw_vua_test, device, args.bert_base_model_dir, args.num_epochs, args.max_len, args.batch_size, args.lr, args.max_grad_norm, args.repr, args.relmodel, args.dropout_rate)

if __name__ == "__main__":
    main()

