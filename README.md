# MR-BERT (Metaphor-Relation BERT)

This repository provides the codes and data for the paper accepted by ACL 2021: [Verb Metaphor Detection via Contextual Relation Learning](https://aclanthology.org/2021.acl-long.327/).

## The main idea
The main idea of this paper is to view metaphor detection as a relation classification problem, i.e., the relation between a target verb and its various contexts.
This idea is consistent with metaphor related theories, and can exploit techniques used in relation modeling.

![image](https://user-images.githubusercontent.com/42745094/135189978-330434e7-3e8b-46a5-8c7a-888ef1078fe5.png)

## Dataset 

|- | # tokens | # unique sent. | % metaphor|
|- | :-: | :-: | :-: |
|VUA-train | 116,622 | 6,323 | 11.2 | 
|VUA-dev | 38,628| 1,550 | 11.6 |
|VUA-Test | 50,175(5,873) | 2,694 | 12.4 |
|MOH-X | 647 | 647 | 48.7 | 
|TroFi | 3,737 | 3,737 | 43.5 |

We mainly conduct experiments on the **VUA** dataset. We use the preprocessed version of the VUA dataset provided by (Gao et al., 2018).
We also report the results on **MOH-X** and **TroFi** in three settings: zero-shot transfer, re-training and fine-tuning.

## Requirements
* pip install -r requirements.txt

* You need to download the bert base model--'bert-base-uncased'

## Run

* Get the embeddings from [here](https://drive.google.com/drive/folders/13_MRmZryGhCf8ngBs57oCMH9KD83ZImu?usp=sharing) and put it to the folder 'data'

* train
   * python main_vua.py --bert_base_model_dir model_dir(the bert base model dir you downloaded)
   * e.g. python main_vua.py --bert_base_model_dir /home/model/bert-base-uncased
    
* test
   * python test_vua.py
    
* you can get the model we saved from [here](https://drive.google.com/drive/folders/1iWrftTDH2If6UO9M-hmc13EwYP1FVfSJ?usp=sharing)

## Update

* We update the integration strategy 'Average' to 'attention'. See the following two files for details:
main_vua_extend.py, cosine_model.py

* python main_vua_extend.py --bert_base_model_dir model_dir

## Citation Information

```bibtex
@inproceedings{song-etal-2021-verb,
    title = "Verb Metaphor Detection via Contextual Relation Learning",
    author = "Song, Wei  and
      Zhou, Shuhui  and
      Fu, Ruiji  and
      Liu, Ting  and
      Liu, Lizhen",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.327",
    doi = "10.18653/v1/2021.acl-long.327",
    pages = "4240--4251",
    abstract = "Correct natural language understanding requires computers to distinguish the literal and metaphorical senses of a word. Recent neu- ral models achieve progress on verb metaphor detection by viewing it as sequence labeling. In this paper, we argue that it is appropriate to view this task as relation classification between a verb and its various contexts. We propose the Metaphor-relation BERT (Mr-BERT) model, which explicitly models the relation between a verb and its grammatical, sentential and semantic contexts. We evaluate our method on the VUA, MOH-X and TroFi datasets. Our method gets competitive results compared with state-of-the-art approaches.",
}
```
