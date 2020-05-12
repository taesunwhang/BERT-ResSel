from collections import defaultdict

BASE_PARAMS = defaultdict(
  # lambda: None,  # Set default value to None.
  # GPU params
  gpu_ids=[0],

  # Input params
  train_batch_size=8,
  eval_batch_size=100,
  virtual_batch_size=32,

  evaluate_candidates_num=10,
  recall_k_list=[1,2,5,10],

  # Training BERT params
  learning_rate=3e-5,
  dropout_keep_prob=0.8,
  num_epochs=10,
  max_gradient_norm=5,

  pad_idx=0,
  max_position_embeddings=100,
  num_hidden_layers=12,
  num_attention_heads=12,
  intermediate_size=3072,
  bert_hidden_dim=768,
  attention_probs_dropout_prob=0.1,
  layer_norm_eps=1e-12,

  # Train Model Config
  task_name="ubuntu",
  do_bert=True,
  do_eot=True,
  max_dialog_len=448,
  max_response_len=64,
  # summation -> 512

  # Need to change to train...(e.g.data dir, config dir, vocab dir, etc.)
  save_dirpath='checkpoints/', # /path/to/checkpoints

  bert_pretrained="bert-base-uncased", # should be defined here
  bert_checkpoint_path="bert-base-uncased-pytorch_model.bin",
  model_type="bert_base_ft",

  load_pthpath="",
  cpu_workers=8,
  tensorboard_step=1000,
  evaluate_print_step=100,
)

DPT_FINETUNING_PARAMS = BASE_PARAMS.copy()
DPT_FINETUNING_PARAMS.update(
  bert_checkpoint_path="bert-post-uncased-pytorch_model.pth", # should be defined here
  model_type="bert_dpt_ft"
)

POST_TRAINING_PARAMS = BASE_PARAMS.copy()
POST_TRAINING_PARAMS.update(
  num_epochs=3,
  # lambda: None,  # Set default value to None.
  # GPU params
  gpu_ids=[0],

  # Input params
  train_batch_size=8,
  virtual_batch_size=512,
  tensorboard_step=100,

  checkpoint_save_step=2500, # virtual_batch -> 10000 step
  model_type="bert_ubuntu_pt",
  data_dir="./data/ubuntu_corpus_v1/ubuntu_post_training.hdf5",
)