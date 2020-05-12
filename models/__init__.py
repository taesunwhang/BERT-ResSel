from models.bert_base_cls import BERTbase
from models.bert_post_training import BertDomainPostTraining


def Model(hparams, *args):
  name_model_map = {
    "bert_base_ft" : BERTbase,
    "bert_dpt_ft" : BERTbase,

    "post_training" : BertDomainPostTraining,
  }

  return name_model_map[hparams.model_type](hparams, *args)