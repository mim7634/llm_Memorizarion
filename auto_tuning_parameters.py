import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, DefaultDataCollator
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import datetime
import torch.nn.functional as F

from train_model_with_snpshot import TRAIN_MODEL

if __name__ == '__main__':
    
    parameter_set_list = [
        ['incremental_learning_embd_56', 4, 8, 56],
        ['incremental_learning_embd_48', 4, 8, 48],
        ['incremental_learning_embd_40', 4, 8, 40],
        ['incremental_learning_embd_32', 4, 8, 32],
        ['incremental_learning_embd_24', 4, 8, 24],
        ['incremental_learning_embd_16', 4, 8, 16],
        ['incremental_learning_embd_8', 4, 8, 8],
    ]

    for param in parameter_set_list:
        n_layer=param[1]
        n_head=param[2]
        n_embd=param[3]
        
        trainer = TRAIN_MODEL(snapshot_interval=500, ppl_stop=False, num_epochs=10000, n_layer=n_layer, n_head=n_head, n_embd=n_embd, model_basa_dir='model/'+param[0])
        trainer.run_training()
