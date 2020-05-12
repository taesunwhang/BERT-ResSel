import os
import torch
import torch.nn as nn

from models.bert import modeling_bert, configuration_bert

class BERTbase(nn.Module):
  def __init__(self, hparams):
    super(BERTbase, self).__init__()
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
      self._bert_model.resize_token_embeddings(self._bert_model.config.vocab_size + 1) # [EOT]

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.bert_hidden_dim, 1)
    )

  def forward(self, batch):
    bert_outputs, _ = self._bert_model(
      batch["anno_sent"],
      token_type_ids=batch["segment_ids"],
      attention_mask=batch["attention_mask"]
    )
    cls_logits = bert_outputs[:,0,:] # bs, bert_output_size
    logits = self._classification(cls_logits) # bs, 1
    logits = logits.squeeze(-1)

    return logits