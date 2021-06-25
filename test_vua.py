# -*- coding: utf-8 -*-
import os
import csv
import ast
from collections import defaultdict
import torch
import random
import logging
from numpy import *
import numpy as np
import torch.nn.functional as F
import argparse
from tokenizers import BertWordPieceTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertConfig
from model import MrBERT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging = logging.getLogger(__name__)

"""
1. Data pre-processing
"""

def load_vua_test():
    """ load  VUA test
    """
    test_svo_labels, test_seq_labels = [], []
    with open('../MrBERT2/data/VUA_final/VUA_test_labels.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            test_seq_labels.append(ast.literal_eval(line[0]))
            test_svo_labels.append(ast.literal_eval(line[1]))
    test_pos= defaultdict(list)
    with open('../MrBERT2/data/VUA_sequence/VUA_seq_formatted_test.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            test_pos[line[0]+'_'+line[1]]=ast.literal_eval(line[4])
    with open('../mlm_vua/embeddings/vua_test_embeddings_ave.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        test_embeddings = []
        for line in lines:
            test_embeddings.append(ast.literal_eval(line[1]))
    with open('../MrBERT2/data/VUA_final/VUA_formatted_test_final.csv', encoding='utf8') as f:
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
            raw_vua_test.append([line[3].strip(), label_seq, s_pos, v_pos, o_pos, int(line[5]), line[0] + '_' + line[1],
                                 test_embeddings[i]])
            i += 1
    random.shuffle(raw_vua_test)
    return raw_vua_test, test_pos


def insert_tag(sentences, s_pos, v_pos, o_pos):
    '''
    insert [subj],[/subj],[verb],[/verb],[obj],[/obj]
    '''
    tokenized_texts = []
    for i in range(len(sentences)):
        sen = sentences[i].split()
        v = sen[v_pos[i]]
        sen[v_pos[i]] = '[verb] ' + v + ' [/verb]'
        if not (s_pos[i] == -1) and not (o_pos[i] == -1):
            s = sen[s_pos[i]]
            sen[s_pos[i]] = '[subj] ' + s + ' [/subj]'
            o = sen[o_pos[i]]
            sen[o_pos[i]] = '[obj] ' + o + ' [/obj]'
        elif not s_pos[i] == -1 and o_pos[i] == -1:
            s = sen[s_pos[i]]
            sen[s_pos[i]] = '[subj] ' + s + ' [/subj]'
        elif s_pos[i] == -1 and not o_pos[i] == -1:
            o = sen[o_pos[i]]
            sen[o_pos[i]] = '[obj] ' + o + ' [/obj]'
        txt = (' '.join(sen)).split()
        tokenized_texts.append(txt)
    return tokenized_texts


def get_inputs(tokenizer, texts, labels0, labels2, max_len, embeddings):
    '''
     encode inputs
    '''
    ids, labels = [], []
    i = 0
    for txt in texts:
        id = [101]
        label = [-100]
        j = 0
        for w in txt:
            enc = tokenizer.encode(w)
            id_w = enc.ids
            id.extend(id_w[1:len(id_w) - 1])
            if w == '[subj]' or w == '[/subj]' or w == '[verb]' or w == '[/verb]' or w == '[obj]' or w == '[/obj]':
                l = [-100]
            else:
                l = [labels0[i][j]]
                if len(enc.tokens) > 3:
                    for t in range(2, len(id_w) - 1):
                        l.append(-100)
                j += 1
            label.extend(l)
        id.append(102)
        label.append(-100)
        assert len(labels0[i]) == (len(label) - label.count(-100))
        assert len(label) == len(id)
        id = id + [0] * (max_len - len(id))
        label = label + [-100] * (max_len - len(label))
        ids.append(id)
        labels.append(label)
        i += 1
    input_ids = torch.tensor([[i for i in id] for id in ids])
    labels = torch.tensor([[i for i in label] for label in labels])
    labels2 = torch.tensor(labels2, dtype=torch.float)
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    # ! 设置 mask_attention
    masks = torch.tensor([[float(i > 0) for i in input_id]
                          for input_id in input_ids])
    return input_ids, labels, labels2, masks, embeddings


def get_results(labels, preds):
    '''
    calculating P R F1-score
    '''
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print("{:15}{:<.3f}".format('accuracy:', accuracy))
    print("{:15}{:<.3f}".format('precision:', precision))
    print("{:15}{:<.3f}".format('recall:', recall))
    print("{:15}{:<.3f}".format('f1', f1))

    return


def evaluation(sen_ids, pos, labels1, preds1, labels2, preds2):

    print('====== VUA VERB ======')
    print(len(labels2))
    get_results(labels2, preds2)

    p_labels, t_labels, pos4_labels, pos4_preds = [], [], [], []
    id = []
    for i in range(len(labels2)):
        if sen_ids[i] not in id:
            id.append(sen_ids[i])
            seq_label = labels1[i][0]
            seq_pred = preds1[i][0]
            seq_pos = pos[sen_ids[i]]
            t_labels.extend(seq_label)
            p_labels.extend(seq_pred)
            assert len(seq_pos) == len(seq_label)
            assert len(seq_label) == len(seq_pred)
            for j in range(len(seq_pos)):
                if 'VERB' == seq_pos[j] or 'NOUN' == seq_pos[j] or 'ADJ' == seq_pos[j] or 'ADV' == seq_pos[j]:
                    pos4_labels.append(seq_label[j])
                    pos4_preds.append(seq_pred[j])
    print('====== ALL  POS ======')
    print(len(t_labels))
    get_results(t_labels, p_labels)
    print('======  4  POS  ======')
    print(len(pos4_labels))
    get_results(pos4_labels, pos4_preds)


def test(test_raw_vua, test_pos, device, model_dir, max_len, repr, integrate, relmodel):
    '''test
    '''
    test_sentences = [r[0] for r in test_raw_vua]

    test_labels0 = [r[1] for r in test_raw_vua]

    test_s_pos = [r[2] for r in test_raw_vua]

    test_v_pos = [r[3] for r in test_raw_vua]

    test_o_pos = [r[4] for r in test_raw_vua]

    test_labels2 = [[r[5]] for r in test_raw_vua]

    test_embeddings = [[r[7]] for r in test_raw_vua]

    test_sen_ids = [r[6] for r in test_raw_vua]

    # if not os.path.exists('./results'):
    #     os.mkdir('./results')

    # headers = ['sentence', 'label']
    # fname = './results/test_vua_seq.csv'
    # with open(fname, 'w')as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(headers)
    #     f_csv.writerows(np.array(test_raw_vua)[:,:7])

    test_tokenized_texts = insert_tag(test_sentences, test_s_pos, test_v_pos, test_o_pos)

    tokenizer = BertWordPieceTokenizer(model_dir + '/vocab.txt', lowercase=True)
    tokenizer.add_special_tokens(['[subj]', '[/subj]', '[verb]', '[/verb]', '[obj]', '[/obj]'])
    config = BertConfig.from_pretrained(model_dir + '/config.json')
    config.num_labels1 = 2
    config.num_labels2 = 2
    
    model = MrBERT(model_dir=model_dir, config=config, device=device, repr=repr, integrate=integrate, relmodel=relmodel)
    checkpoint = torch.load('./model/pytorch_model.bin', map_location=device)
    model.load_state_dict(checkpoint)
    # model = MrBERT.from_pretrained('./model', config=config, device=device, model_dir=model_dir, repr=repr,
    #                                integrate=integrate, relmodel=relmodel)

    model.to(device)

    # ! get inputs
    test_input_ids, test_labels1, test_labels2, test_masks, test_embeddings = get_inputs(tokenizer,
                                                                                         test_tokenized_texts,
                                                                                         test_labels0, test_labels2,
                                                                                         max_len, test_embeddings)
    test_data = TensorDataset(test_input_ids, test_masks, test_labels1, test_labels2, test_embeddings)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # ! Test
    model.eval()
    preds1, labels1, preds2, labels2, t_labels1, p_labels1, t_labels2, p_labels2 = [], [], [], [], [], [], [], []
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels1, b_labels2, b_embeddings = batch

        with torch.no_grad():
            outputs1, outputs2 = model(input_ids=b_input_ids, token_type_ids=None,
                                       attention_mask=b_input_mask, labels1=b_labels1, labels2=b_labels2,
                                       embeddings=b_embeddings)
            tmp_eval_loss1, logits1 = outputs1[:2]
            tmp_eval_loss2, logits2 = outputs2[:2]

        values1, logits1 = torch.max(F.softmax(logits1, dim=-1), dim=-1)[:2]

        logits2 = logits2.view(-1)
        if logits2 > 0.5:
            logits2 = torch.Tensor([1])
        else:
            logits2 = torch.Tensor([0])

        ture_labels1 = b_labels1[0]
        logits1 = logits1[0]
        ture_labels2 = b_labels2[0][0]
        logits2 = logits2[0]

        t_labels1.append(ture_labels1)
        p_labels1.append(logits1)
        t_labels2.append(ture_labels2)
        p_labels2.append(logits2)

        # ! detach的方法，将variable参数从网络中隔离开，不参与参数更新
        logits1 = logits1.detach().cpu().numpy()
        logits2 = logits2.detach().cpu().numpy()
        ture_labels1 = ture_labels1.cpu().numpy()
        ture_labels2 = ture_labels2.cpu().numpy()

        preds1.extend(logits1)
        labels1.extend(ture_labels1)
        preds2.append(int(logits2))
        labels2.append(int(ture_labels2))

    # ! 计算评估值
    results = []
    for i in range(len(p_labels1)):
        pre, label = [], []
        for j in range(len(t_labels1[i])):
            if not t_labels1[i][j] == -100:
                label.append(int(t_labels1[i][j]))
                pre.append(int(p_labels1[i][j]))
        results.append([label, pre, int(t_labels2[i]), int(p_labels2[i])])

    # 打印信息
    print("--- Test ---")
    evaluation(test_sen_ids, test_pos, np.array(results)[:,:1], np.array(results)[:,1:2], labels2, preds2)

    # if val_f12 >= val_max_f1:
    #     headers = ['labels1','preds1','labels2','preds2']
    #     fname = './results/vua_preds_seq.csv'
    #     with open(fname, 'w')as f:
    #         f_csv = csv.writer(f)
    #         f_csv.writerow(headers)
    #         f_csv.writerows(list(results))


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
    parser.add_argument('--repr', default='average', type=str, required=False, choices=['tag', 'average'],
                        help='获取表示形式：tag/average')
    parser.add_argument('--integrate', default='average', type=str, required=False,
                        choices=['average', 'maxout', 'concat'], help='整合策略：average/maxout/concat')
    parser.add_argument('--relmodel', default='bilinear', type=str, required=False,
                        choices=['linear', 'bilinear', 'nt'], help='模型：linear/bilinear/nt（neural tensor）')
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

    raw_vua_test, test_pos = load_vua_test()

    test(raw_vua_test, test_pos, device, args.bert_base_model_dir, args.max_len, args.repr, args.integrate, args.relmodel)


if __name__ == "__main__":
    main()
