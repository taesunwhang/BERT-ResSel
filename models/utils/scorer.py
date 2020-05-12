import numpy as np

def calculate_candidates_ranking(prediction, ground_truth, eval_candidates_num=10):
  total_num_split = len(ground_truth) / eval_candidates_num

  pred_split = np.split(prediction, total_num_split)
  gt_split = np.split(np.array(ground_truth), total_num_split)
  orig_rank_split = np.split(np.tile(np.arange(0, eval_candidates_num), int(total_num_split)), total_num_split)
  stack_scores = np.stack((gt_split, pred_split, orig_rank_split), axis=-1)

  rank_by_pred_l = []
  for i, stack_score in enumerate(stack_scores):
    rank_by_pred = sorted(stack_score, key=lambda x: x[1], reverse=True)
    rank_by_pred = np.stack(rank_by_pred, axis=-1)
    rank_by_pred_l.append(rank_by_pred[0])

  return np.array(rank_by_pred_l)

def logits_recall_at_k(rank_by_pred, k_list=[1, 2, 5, 10]):
  # 1 dialog, 10 response candidates ground truth 1 or 0
  # prediction_score : [batch_size]
  # target : [batch_size] e.g. 1 0 0 0 0 0 0 0 0 0
  # e.g. batch : 100 -> 100/10 = 10

  num_correct = np.zeros([rank_by_pred.shape[0], len(k_list)])

  pos_index = []
  for sorted_score in rank_by_pred:
    for p_i, score in enumerate(sorted_score):
      if int(score) == 1:
        pos_index.append(p_i)
  index_dict = dict()
  for i, p_i in enumerate(pos_index):
    index_dict[i] = p_i

  for i, p_i in enumerate(pos_index):
    for j, k in enumerate(k_list):
      if p_i + 1 <= k:
        num_correct[i][j] += 1

  return np.sum(num_correct, axis=0), pos_index

def logits_mrr(rank_by_pred):
  pos_index = []
  for sorted_score in rank_by_pred:
    for p_i, score in enumerate(sorted_score):
      if int(score) == 1:
        pos_index.append(p_i)

  # print("pos_index", pos_index)
  mrr = []
  for i, p_i in enumerate(pos_index):
    mrr.append(1 / (p_i + 1))

  # print(mrr)

  return np.sum(mrr)