import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Model
from data.dataset import ResponseSelectionDataset
from models.utils.checkpointing import load_checkpoint
from models.utils.scorer import calculate_candidates_ranking, logits_mrr, logits_recall_at_k

class Evaluation(object):
	def __init__(self, hparams, model=None, split = "test"):

		self.hparams = hparams
		self.model = model
		self._logger = logging.getLogger(__name__)
		self.device = (torch.device("cuda", self.hparams.gpu_ids[0])
									 if self.hparams.gpu_ids[0] >= 0 else torch.device("cpu"))
		self.split = split
		print("Evaluation Split :", self.split)
		do_valid, do_test = False, False
		if split == "valid":
			do_valid = True
		else:
			do_test = True
		self._build_dataloader(do_valid=do_valid, do_test=do_test)
		self._dataloader = self.valid_dataloader if split == 'valid' else self.test_dataloader

		if model is None:
			print("No pre-defined model!")
			self._build_model()

	def _build_dataloader(self, do_valid=False, do_test=False):

		if do_valid:
			self.valid_dataset = ResponseSelectionDataset(
				self.hparams,
				split="valid",
			)
			self.valid_dataloader = DataLoader(
				self.valid_dataset,
				batch_size=self.hparams.eval_batch_size,
				num_workers=self.hparams.cpu_workers,
				drop_last=False,
			)

		if do_test:
			self.test_dataset = ResponseSelectionDataset(
				self.hparams,
				split="test",
			)

			self.test_dataloader = DataLoader(
				self.test_dataset,
				batch_size=self.hparams.eval_batch_size,
				num_workers=self.hparams.cpu_workers,
				drop_last=False,
			)

	def _build_model(self):
		self.model = Model(self.hparams)
		self.model = self.model.to(self.device)
		# Use Multi-GPUs
		if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
			self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

	def run_evaluate(self, evaluation_path):
		self._logger.info("Evaluation")
		model_state_dict, optimizer_state_dict = load_checkpoint(evaluation_path)

		if isinstance(self.model, nn.DataParallel):
			self.model.module.load_state_dict(model_state_dict)
		else:
			self.model.load_state_dict(model_state_dict)

		k_list = self.hparams.recall_k_list
		total_mrr = 0
		total_examples, total_correct = 0, 0
		self.model.eval()
		with torch.no_grad():
			for batch_idx, batch in enumerate(tqdm(self._dataloader)):
				buffer_batch = batch.copy()
				for key in batch:
					buffer_batch[key] = batch[key].to(self.device)

				logits = self.model(buffer_batch)
				pred = torch.sigmoid(logits).to("cpu").tolist()  # bs

				rank_by_pred = calculate_candidates_ranking(np.array(pred), np.array(buffer_batch["label"].to("cpu").tolist()),
																										self.hparams.evaluate_candidates_num)
				num_correct, pos_index = logits_recall_at_k(rank_by_pred, k_list)

				total_mrr += logits_mrr(rank_by_pred)

				total_correct = np.add(total_correct, num_correct)
				total_examples = (batch_idx + 1) * rank_by_pred.shape[0]

				recall_result = ""
				if (batch_idx + 1) % self.hparams.evaluate_print_step == 0:
					for i in range(len(k_list)):
						recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % ((total_correct[i] / total_examples) * 100)
					else:
						print("%d[th] | %s | MRR : %.3f" % (batch_idx + 1, recall_result, float(total_mrr / total_examples)))
					self._logger.info("%d[th] | %s | MRR : %.3f" % (batch_idx + 1, recall_result, float(total_mrr / total_examples)))

			avg_mrr = float(total_mrr / total_examples)
			recall_result = ""

			for i in range(len(k_list)):
				recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % ((total_correct[i] / total_examples) * 100)
			self._logger.info(recall_result)
			self._logger.info("MRR: %.4f" % avg_mrr)