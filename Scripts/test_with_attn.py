import os
import tarfile
import requests
import pandas as pd
from io import BytesIO
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import *
import wandb
from load_data import *
from train_with_attn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = AttnSeq2Seq(
            input_dim=len(src_vocab.char2idx),
            output_dim=len(trg_vocab.char2idx)+1,
            embed_dim=256,
            hidden_dim=256,
            num_layers=2,
            cell_type='RNN',
            dropout=0.2,
            device=device,
            go_idx = trg_vocab.char2idx['<go>'],
            stop_idx = trg_vocab.char2idx['<stop>'],
            vocab = trg_vocab
        ).to(device)
best_model.load_state_dict(torch.load('/kaggle/input/attn_model_2/pytorch/default/1/attn_model.pth',  weights_only = True))
best_model.eval()
evaluate_attn(best_model, test_loader, device)