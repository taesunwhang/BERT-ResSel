import os
import argparse
import collections
import logging
from datetime import datetime

from config.hparams import *
from data.data_utils import InputExamples
from train import ResponseSelection
from evaluation import Evaluation
from post_train import BERTDomainPostTraining

PARAMS_MAP = {
  # fine-tuning (ft)
  "bert_base_ft" : BASE_PARAMS,
  "bert_dpt_ft" : DPT_FINETUNING_PARAMS,

  # post-training (pt)
  "bert_ubuntu_pt" : POST_TRAINING_PARAMS,
}

MODEL = {
  "fine_tuning" : ResponseSelection,
  "post_training" : BERTDomainPostTraining
}

def init_logger(path:str):
  if not os.path.exists(path):
      os.makedirs(path)
  logger = logging.getLogger()
  logger.handlers = []
  logger.setLevel(logging.DEBUG)
  debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
  debug_fh.setLevel(logging.DEBUG)

  info_fh = logging.FileHandler(os.path.join(path, "info.log"))
  info_fh.setLevel(logging.INFO)

  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)

  info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
  debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

  ch.setFormatter(info_formatter)
  info_fh.setFormatter(info_formatter)
  debug_fh.setFormatter(debug_formatter)

  logger.addHandler(ch)
  logger.addHandler(debug_fh)
  logger.addHandler(info_fh)

  return logger

def train_model(args):
    hparams = PARAMS_MAP[args.model]
    hparams["root_dir"] = args.root_dir
    hparams["bert_pretrained_dir"] = args.bert_pretrained_dir
    hparams["bert_pretrained"] = args.bert_pretrained
    hparams["model_type"] = args.model

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(hparams["root_dir"], args.model, args.train_type, "%s/" % timestamp)
    logger = init_logger(root_dir)
    logger.info("Hyper-parameters: %s" % str(hparams))
    hparams["root_dir"] = root_dir

    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
    model = MODEL[args.train_type](hparams)
    model.train()

def evaluate_model(args):
  hparams = PARAMS_MAP[args.model]

  hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

  model = Evaluation(hparams)
  model.run_evaluate(args.evaluate)

if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(description="Bert / Response Selection (PyTorch)")
  arg_parser.add_argument("--model", dest="model", type=str,
                          default="bert_base",
                          help="Model Name")
  arg_parser.add_argument("--root_dir", dest="root_dir", type=str,
                          default="./results",
                          help="model train logs, checkpoints")
  arg_parser.add_argument("--data_dir", dest="data_dir", type=str,
                          default="./data/ubuntu_corpus_v1/%s_%s.pkl",
                          help="ubuntu corpus v1 pkl path") # ubuntu_train.pkl, ubuntu_valid_pkl, ubuntu_test.pkl
  arg_parser.add_argument("--bert_pretrained_dir", dest="bert_pretrained_dir", type=str,
                          default="./resources",
                          help="bert pretrained directory")
  arg_parser.add_argument("--bert_pretrained", dest="bert_pretrained", type=str,
                          default="bert-post-uncased",
                          help="bert pretrained directory")  # bert-base-uncased, bert-post-uncased -> under bert_pretrained_dir
  arg_parser.add_argument("--train_type", dest="train_type", type=str,
                          default="fine_tuning",
                          help="Train type") # fine_tuning, post_training
  arg_parser.add_argument("--evaluate", dest="evaluate", type=str,
                          help="Evaluation Checkpoint", default="")

  args = arg_parser.parse_args()
  if args.evaluate:
    evaluate_model(args)
  else:
    train_model(args)