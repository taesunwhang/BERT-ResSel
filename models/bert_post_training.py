import os
import torch.nn as nn

from models.bert import modeling_bert, configuration_bert

class BertDomainPostTraining(nn.Module):
  def __init__(self, hparams):
    super(BertDomainPostTraining, self).__init__()
    self.hparams = hparams
    bert_config = configuration_bert.BertConfig.from_pretrained(
      os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
                   "%s-config.json" % self.hparams.bert_pretrained),
    )
    self._bert_model = modeling_bert.BertModel.from_pretrained(
      os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained, self.hparams.bert_checkpoint_path),
      config=bert_config
    )

    if self.hparams.do_eot and self.hparams.model_type == "bert_base_ft":
      self._bert_model.resize_token_embeddings(self._bert_model.config.vocab_size + 1)  # [EOT]

  def forward(self, batch):
    bert_outputs = self._bert_model(
      input_ids=batch["input_ids"],
      token_type_ids=batch["token_type_ids"],
      attention_mask=batch["attention_mask"],
      masked_lm_labels=batch["masked_lm_labels"],
      next_sentence_label=batch["next_sentence_labels"]
    )
    mlm_loss, nsp_loss, prediction_scores, seq_relationship_score = bert_outputs[:4]

    return mlm_loss, nsp_loss