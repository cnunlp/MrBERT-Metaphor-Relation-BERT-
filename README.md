# MrBERT

This is the official code for the ACL 2021 paper:[Verb Metaphor Detection via Contextual Relation Learning](hhh).
## Dataset 
｜--- | # tokens | # unique sent. | % metaphor｜
｜-- | :-: | :-: | :-: ｜
VUA-train | 116,622 | 6,323 | 11.2 | 
VUA-dev | 38,628| 1,550 | 11.6 |
VUA-Test | 50,175(5,873) | 2,694 | 12.4 |
MOH-X | 647 | 647 | 48.7 | 
TroFi | 3,737 | 3,737 | 43.5 |
We mainly conduct experiments on the **VUA** dataset.We use the preprocessed version of the VUA dataset provided by (Gao et al., 2018).
We also report the results on **MOH-X** and **TroFi** in three settings:zero-shot transfer, re-training and fine-tuning.
## Requirements
* pip install -r requirements.txt

* You need to download the bert base model--'bert-base-uncased'

## Run

* Get the data from [here](https://drive.google.com/drive/folders/13_MRmZryGhCf8ngBs57oCMH9KD83ZImu?usp=sharing)

* python main_vua.py --bert_base_model_dir model_dir(the bert base model dir you downloaded)
    * e.g. python main_vua.py --bert_base_model_dir /home/model/bert-base-uncased
    
* python test_vua.py --bert_base_model_dir model_dir(the bert base model dir you downloaded)
    * e.g. python test_vua.py --bert_base_model_dir /home/model/bert-base-uncased
    
* you can get the model we saved from [here](hhh)
