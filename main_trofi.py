import argparse
import ast
import csv
import os
import torch
import random
import logging
from numpy import *
import numpy as np
import torch.nn.functional as F
from tokenizers import BertWordPieceTokenizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from model import MrBERT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging = logging.getLogger(__name__)

def load_trofi():
    test_svo_labels ,test_seq_labels = [], []
    with open('../MrBERT/data/TroFi/trofi_labels.txt', encoding='utf8') as f:
        for line in f:
            test_svo_labels.append(line.split())
    with open('../MrBERT/data/embeddings_trofi/trofi_embeddings_ave.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        test_embeddings = []
        for line in lines:
            test_embeddings.append(ast.literal_eval(line[1]))
    raw_trofi = []
    with open('../MrBERT/data/TroFi/TroFi_formatted_all3737.csv', encoding='utf8') as f:
        lines = csv.reader(f)
        next(lines)
        i = 0
        for line in lines:
            sen = line[1].split()
            v_pos = int(line[2])
            s_pos = -1
            o_pos = -1
            if '1' in test_svo_labels[i]:
                s_pos = test_svo_labels[i].index('1')
            if '3' in test_svo_labels[i]:
                o_pos = test_svo_labels[i].index('3')
            label_seq = [0] * len(sen)
            label_seq[v_pos] = int(line[3])
            assert (len(label_seq) == len(sen))
            assert len(test_svo_labels[i]) == len(sen)
            raw_trofi.append([line[1].strip(), label_seq, s_pos, v_pos, o_pos, int(line[3]), test_embeddings[i]])
            i += 1
    random.shuffle(raw_trofi)
    return raw_trofi

def insert_tag(sentences, s_pos, v_pos, o_pos):
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
    ids = []
    labels = []
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

def get_kfold_data(k, i, raw_trofi):
    fold_size = len(raw_trofi) // k

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        val_raw_trofi = raw_trofi[val_start:val_end]
        tr_raw_trofi = raw_trofi[0:val_start]+raw_trofi[val_end:]
    else:
        val_raw_trofi= raw_trofi[val_start:]
        tr_raw_trofi = raw_trofi[0:val_start]

    return tr_raw_trofi, val_raw_trofi

def traink(tr_raw_trofi, val_raw_trofi, TOTAL_EPOCHS, device, model_dir, max_len, batch_size, learning_rate, repr, integrate, relmodel, operation):
    # ! get data
    tr_sentences = [r[0] for r in tr_raw_trofi]
    val_sentences = [r[0] for r in val_raw_trofi]

    tr_labels0 = [r[1] for r in tr_raw_trofi]
    val_labels0 = [r[1] for r in val_raw_trofi]

    tr_s_pos = [r[2] for r in tr_raw_trofi]
    val_s_pos = [r[2] for r in val_raw_trofi]

    tr_v_pos = [r[3] for r in tr_raw_trofi]
    val_v_pos = [r[3] for r in val_raw_trofi]

    tr_o_pos = [r[4] for r in tr_raw_trofi]
    val_o_pos = [r[4] for r in val_raw_trofi]

    tr_labels2 = [[r[5]] for r in tr_raw_trofi]
    val_labels2 = [[r[5]] for r in val_raw_trofi]

    tr_embeddings = [[r[6]] for r in tr_raw_trofi]
    val_embeddings = [[r[6]] for r in val_raw_trofi]

    tr_tokenized_texts = insert_tag(tr_sentences, tr_s_pos, tr_v_pos, tr_o_pos)

    val_tokenized_texts = insert_tag(val_sentences, val_s_pos, val_v_pos, val_o_pos)

    tokenizer = BertWordPieceTokenizer('./vocab.txt',lowercase=True)
    tokenizer.add_special_tokens(['[subj]','[/subj]', '[verb]','[/verb]','[obj]','[/obj]'])
    config = BertConfig.from_pretrained(model_dir + '/config.json')
    config.num_labels1 = 2
    config.num_labels2 = 2

    # ! train or finetune
    if operation == 'train':
        model = MrBERT.from_pretrained(model_dir, config=config, bert_model_dir=model_dir, device=device, repr=repr,
                                       integrate=integrate, relmodel=relmodel)
    elif operation == 'finetune':
        model = MrBERT(bert_model_dir=model_dir, config=config, device=device, repr=repr, integrate=integrate,
                       relmodel=relmodel)
        checkpoint = torch.load('./model/pytorch_model.bin', map_location=device)
        model.load_state_dict(checkpoint)

    model.to(device)

    # ! get inputs
    tr_input_ids, tr_labels1, tr_labels2, tr_masks, tr_embeddings = get_inputs(tokenizer, tr_tokenized_texts, tr_labels0, tr_labels2, max_len, tr_embeddings)
    val_input_ids, val_labels1, val_labels2, val_masks, val_embeddings = get_inputs(tokenizer, val_tokenized_texts, val_labels0, val_labels2, max_len, val_embeddings)

    # ! get dataloader
    train_data = TensorDataset(tr_input_ids, tr_masks, tr_labels1, tr_labels2, tr_embeddings)
    train_loader = DataLoader(train_data,  batch_size=batch_size,shuffle=True)

    val_data = TensorDataset(val_input_ids, val_masks, val_labels1, val_labels2, val_embeddings)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # ! optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate)

    t_total = len(tr_input_ids)/batch_size*TOTAL_EPOCHS+1
    num_warmup_steps = int(t_total/10)*2
    logging.info('t_total: %d warmup: %d' % (t_total, num_warmup_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

    val_accs1, val_accs2, val_ps1, val_ps2, val_rs1, val_rs2, val_f1s1, val_f1s2, results1,results2  = [],[],[],[],[],[],[],[],[],[]
    for epoch in range(TOTAL_EPOCHS):
        print('===== Start training: epoch {} ====='.format(epoch + 1))

        model.train()
        tr_loss = 0
        nb_tr_steps = 0

        # ! training
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels1, b_labels2, b_embeddings = batch

            outputs1,outputs2 = model(input_ids=b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels1=b_labels1, labels2=b_labels2, embeddings = b_embeddings)

            loss = outputs1[0] + outputs2[0]

            loss.backward()

            tr_loss += float(loss.item())

            nb_tr_steps += 1

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

        print("\nEpoch {} of training loss: {}".format(epoch + 1, tr_loss / nb_tr_steps))

        # ! Validation
        model.eval()
        nb_eval_steps = 0

        preds1, labels1, preds2, labels2 = [], [], [], []
        for step, batch in enumerate(val_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels1, b_labels2, b_embeddings = batch
            with torch.no_grad():
                outputs1, outputs2 = model(input_ids=b_input_ids, token_type_ids=None,
                                           attention_mask=b_input_mask, labels1=b_labels1, labels2=b_labels2, embeddings=b_embeddings)
                tmp_eval_loss2, logits2 = outputs2[:2]

            logits2 = logits2.view(-1)
            if logits2 > 0.5:
                logits2 = torch.Tensor([1])
            else:
                logits2 = torch.Tensor([0])

            logits2 = logits2.detach().cpu().numpy()
            ture_labels2 = b_labels2[0].cpu().numpy()

            preds2.append(logits2)
            labels2.append(ture_labels2)
            nb_eval_steps += 1

        val_preds2 = np.array(preds2)
        val_labels2 = np.array(labels2)

        # ! evaluation
        val_accuracy2, val_precision2, val_recall2, val_f12 = evaluation(val_labels2, val_preds2)

        val_accs2.append(val_accuracy2)
        val_ps2.append(val_precision2)
        val_rs2.append(val_recall2)
        val_f1s2.append(val_f12)

    print("===== Train Finished =====\n")
    index=val_f1s2.index(max(val_f1s2))
    print("{:15}{:<}".format("max epoch", index+1))
    print("{:15}{:<.3f}".format("accuracy", val_accs2[index]))
    print("{:15}{:<.3f}".format("precision", val_ps2[index]))
    print("{:15}{:<.3f}".format("recall", val_rs2[index]))
    print("{:15}{:<.3f}".format("f1", val_f1s2[index]))

    return val_accs2[index], val_ps2[index], val_rs2[index], val_f1s2[index]

def k_fold(k, raw_trofi, num_epochs, device, model_dir, max_len, batch_size, lr, repr, integrate, relmodel, operation):
    val_acc_sum, val_p_sum = 0, 0
    val_r_sum, val_f1_sum = 0, 0
    for i in range(k):
        print('*' * 25,  i + 1, 'fold', '*' * 25)
        data = get_kfold_data(k, i, raw_trofi)

        val_acc, val_p, val_r, val_f1 = traink( *data, num_epochs, device, model_dir, max_len, batch_size, lr, repr, integrate, relmodel, operation)

        val_acc_sum += val_acc
        val_p_sum += val_p
        val_r_sum += val_r
        val_f1_sum += val_f1

    print('\n', '#' * 10, '10-flod results of TroFi', '#' * 10)

    print('average accuracy:{:.3f}, average precision:{:.3f}'.format(val_acc_sum / k, val_p_sum / k))
    print('average recall:{:.3f}, average f1:{:.3f}'.format(val_r_sum / k, val_f1_sum / k))

    return

def test(test_raw_trofi, device, model_dir, max_len, repr, integrate, relmodel):
    test_sentences = [r[0] for r in test_raw_trofi]

    test_labels0 = [r[1] for r in test_raw_trofi]

    test_s_pos = [r[2] for r in test_raw_trofi]

    test_v_pos = [r[3] for r in test_raw_trofi]

    test_o_pos = [r[4] for r in test_raw_trofi]

    test_labels2 = [[r[5]] for r in test_raw_trofi]

    test_embeddings = [[r[6]] for r in test_raw_trofi]

    test_tokenized_texts= insert_tag(test_sentences, test_s_pos, test_v_pos, test_o_pos)

    tokenizer = BertWordPieceTokenizer('./vocab.txt',lowercase=True)
    tokenizer.add_special_tokens(['[subj]','[/subj]', '[verb]','[/verb]','[obj]','[/obj]'])
    config = BertConfig.from_pretrained(model_dir + '/config.json')
    config.num_labels1 = 2
    config.num_labels2 = 2
    model = MrBERT(bert_model_dir=model_dir, config=config, device=device, repr=repr, integrate=integrate, relmodel=relmodel)
    checkpoint = torch.load('./model/pytorch_model.bin', map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    test_input_ids, test_labels1,test_labels2, test_masks, test_embeddings = get_inputs(tokenizer, test_tokenized_texts, test_labels0, test_labels2, max_len, test_embeddings)

    test_data = TensorDataset(test_input_ids, test_masks, test_labels1, test_labels2, test_embeddings)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # ! Test
    preds1, labels1, preds2, labels2, t_labels1, p_labels1, t_labels2, p_labels2 = [], [], [], [], [], [], [], []
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels1, b_labels2, b_embeddings = batch

        with torch.no_grad():
            outputs1, outputs2 = model(input_ids=b_input_ids, token_type_ids=None,
                                                 attention_mask=b_input_mask, labels1=b_labels1, labels2=b_labels2, embeddings=b_embeddings)
            tmp_eval_loss2, logits2 = outputs2[:2]

        logits2 = logits2.view(-1)
        if logits2 > 0.5:
            logits2 = torch.Tensor([1])
        else:
            logits2 = torch.Tensor([0])

        ture_labels2 = b_labels2[0][0]
        logits2 = logits2[0]

        t_labels2.append(ture_labels2)
        p_labels2.append(logits2)

        logits2 = logits2.detach().cpu().numpy()
        ture_labels2 = ture_labels2.cpu().numpy()

        preds2.append(logits2)
        labels2.append(ture_labels2)

    preds2 = np.array(preds2)
    labels2 = np.array(labels2)

    # 打印信息
    print("--- Test ---")
    eval_accuracy2, eval_precision2, eval_recall2, eval_f12 = evaluation(labels2, preds2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False)
    parser.add_argument('--seed', default=4, type=int, required=False)
    parser.add_argument('--bert_base_model_dir', type=str, required=False)
    parser.add_argument('--max_len', default=130, type=int, required=False)
    parser.add_argument('--kfold', default=10, type=int, required=False)
    parser.add_argument('--batch_size', default=16, type=int, required=False)
    parser.add_argument('--lr', default=5e-5, type=float, required=False)
    parser.add_argument('--num_epochs', default=10, type=int, required=False)
    parser.add_argument('--repr', default='average', type=str, required=False, choices=['tag', 'average'])
    parser.add_argument('--integrate', default='average', type=str, required=False, choices=['average', 'maxout', 'concat'])
    parser.add_argument('--relmodel', default='bilinear', type=str, required=False, choices=['linear', 'bilinear', 'nt'])
    parser.add_argument('--operation', default='train', type=str, required=False, choices=['train', 'finetune', 'test'])
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_trofi = load_trofi()

    if args.operation == 'test':
        test(raw_trofi, device, './model', args.max_len, args.repr, args.integrate, args.relmodel)
    elif args.operation == 'finetune':
        k_fold(args.kfold, raw_trofi, args.num_epochs, device, './model', args.max_len, args.batch_size, args.lr, args.repr, args.integrate, args.relmodel, args.operation)
    else:
        k_fold(args.kfold, raw_trofi, args.num_epochs, device, args.bert_base_model_dir, args.max_len, args.batch_size, args.lr, args.repr, args.integrate, args.relmodel, args.operation)
if __name__ == "__main__":
    main()

