# MrBERT

Code for the paper "Verb Metaphor Detection via Contextual Relation Learning".

## Environment

* pip install -r requirements.txt
* you need to download the bert base model--'bert-base-uncased'

## Run
* python main_vua.py --bert_base_model_dir python main_vua.py --bert_base_model_dir model_dir(the bert base model dir you downloaded)
  * e.g. python main_vua.py --bert_base_model_dir python main_vua.py --bert_base_model_dir /home/model/bert-base-uncased
* python test_vua.py--bert_base_model_dir python main_vua.py --bert_base_model_dir model_dir(the bert base model dir you downloaded)
  * e.g. python test_vua.py --bert_base_model_dir python main_vua.py --bert_base_model_dir /home/model/bert-base-uncased