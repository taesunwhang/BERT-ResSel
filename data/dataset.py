import os
import torch
import pickle
import h5py
import numpy as np

from models.bert import tokenization_bert
from torch.utils.data import Dataset

class ResponseSelectionDataset(Dataset):
    """
    A full representation of VisDial v1.0 (train/val/test) dataset. According
    to the appropriate split, it returns dictionary of question, image,
    history, ground truth answer, answer options, dense annotations etc.
    """
    def __init__(
        self,
        hparams,
        split: str = "",
    ):
        super().__init__()

        self.hparams = hparams
        self.split = split

        # read pkls -> Input Examples
        self.input_examples = []
        with open(hparams.data_dir % (hparams.task_name, split), "rb") as pkl_handle:
          while True:
            try:
              self.input_examples.append(pickle.load(pkl_handle))
              if len(self.input_examples) % 100000 == 0:
                print("%d examples has been loaded!" % len(self.input_examples))
            except EOFError:
              break

        print("total %s examples" % split, len(self.input_examples))

        bert_pretrained_dir = os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained)
        print(bert_pretrained_dir)
        self._bert_tokenizer = tokenization_bert.BertTokenizer(
          vocab_file=os.path.join(bert_pretrained_dir, "%s-vocab.txt" % self.hparams.bert_pretrained))

        # End of Turn Token
        if self.hparams.do_eot:
          self._bert_tokenizer.add_tokens(["[EOT]"])

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, index):
        # Get Input Examples
        """
        InputExamples
          self.utterances = utterances
          self.response = response
          self.label
        """

        anno_sent, segment_ids, attention_mask = self._annotate_sentence(self.input_examples[index])

        current_feature = dict()
        current_feature["anno_sent"] = torch.tensor(anno_sent).long()
        current_feature["segment_ids"] = torch.tensor(segment_ids).long()
        current_feature["attention_mask"] = torch.tensor(attention_mask).long()
        current_feature["label"] = torch.tensor(self.input_examples[index].label).float()

        return current_feature

    def _annotate_sentence(self, example):

      dialog_context = []
      if self.hparams.do_eot:
        for utt in example.utterances:
          dialog_context.extend(utt + ["[EOT]"])
      else:
        for utt in example.utterances:
          dialog_context.extend(utt)

      # Set Dialog Context length to 280, Response length to 40
      dialog_context, response = self._max_len_trim_seq(dialog_context, example.response)

      # dialog context
      dialog_context = ["[CLS]"] + dialog_context + ["[SEP]"]
      segment_ids = [0] * self.hparams.max_dialog_len
      attention_mask = [1] * len(dialog_context)

      while len(dialog_context) < self.hparams.max_dialog_len:
        dialog_context.append("[PAD]")
        attention_mask.append(0)

      assert len(dialog_context) == len(segment_ids) == len(attention_mask)

      response = response + ["[SEP]"]
      segment_ids.extend([1] * len(response))
      attention_mask.extend([1] * len(response))

      while len(response) < self.hparams.max_response_len:
        response.append("[PAD]")
        segment_ids.append(0)
        attention_mask.append(0)

      dialog_response = dialog_context + response

      # print(segment_ids)
      # print(attention_mask)
      # print(len(dialog_response), len(segment_ids), len(attention_mask))

      assert len(dialog_response) == len(segment_ids) == len(attention_mask)
      anno_sent = self._bert_tokenizer.convert_tokens_to_ids(dialog_response)

      return anno_sent, segment_ids, attention_mask

    def _max_len_trim_seq(self, dialog_context, response):
      while len(dialog_context) > self.hparams.max_dialog_len - 2:
        dialog_context.pop(0) # from the front

      while len(response) > self.hparams.max_response_len - 1:
        response.pop() # from the back

      return dialog_context, response

class BertPostTrainingDataset(Dataset):
  """
  A full representation of VisDial v1.0 (train/val/test) dataset. According
  to the appropriate split, it returns dictionary of question, image,
  history, ground truth answer, answer options, dense annotations etc.
  """

  def __init__(
      self,
      hparams,
      split: str = "",
  ):
    super().__init__()

    self.hparams = hparams
    self.split = split

    with h5py.File(self.hparams.data_dir, "r") as features_hdf:
      self.feature_keys = list(features_hdf.keys())
      self.num_instances = np.array(features_hdf.get("next_sentence_labels")).shape[0]
    print("total %s examples : %d" % (split, self.num_instances))

  def __len__(self):
    return self.num_instances

  def __getitem__(self, index):
    # Get Input Examples
    """
    InputExamples
      self.utterances = utterances
      self.response = response
      self.label
    """
    features = self._read_hdf_features(index)
    anno_masked_lm_labels = self._anno_mask_inputs(features["masked_lm_ids"], features["masked_lm_positions"])
    curr_features = dict()
    for feat_key in features.keys():
      curr_features[feat_key] = torch.tensor(features[feat_key]).long()
    curr_features["masked_lm_labels"] = torch.tensor(anno_masked_lm_labels).long()
    return curr_features

  def _read_hdf_features(self, index):
    features = {}
    with h5py.File(self.hparams.data_dir, "r") as features_hdf:
      for f_key in self.feature_keys:
        features[f_key] = features_hdf[f_key][index]

    return features

  def _anno_mask_inputs(self, masked_lm_ids, masked_lm_positions, max_seq_len=320):
    # masked_lm_ids -> labels
    anno_masked_lm_labels = [-1] * max_seq_len

    for pos, label in zip(masked_lm_positions, masked_lm_ids):
      if pos == 0: continue
      anno_masked_lm_labels[pos] = label

    return anno_masked_lm_labels