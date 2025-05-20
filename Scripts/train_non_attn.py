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
from load_data import *
import wandb

def train(config=None):
    with wandb.init(project="DA6401_A3",settings=wandb.Settings(start_method="thread",_disable_stats=True), config = config) as run:
        config = run.config
        
        # Get dataloaders and vocabularies
        train_loader, dev_loader, _, src_vocab, trg_vocab = get_dataloaders(
            batch_size=config.batch_size
        )
        go_idx = trg_vocab.char2idx['<go>']
        stop_idx = trg_vocab.char2idx['stop']
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = Seq2Seq(
            input_dim=len(src_vocab.char2idx),
            output_dim=len(trg_vocab.char2idx),
            embed_dim=config.embedding_size,
            hidden_dim=config.hidden_size,
            num_layers=config.num_layers,
            cell_type=config.cell_type,
            dropout=config.dropout,
            device=device,
            go_idx = go_idx,
            stop_idx = stop_idx,
            vocab = trg_vocab
        ).to(device)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        # Training loop
        for epoch in range(15):  # Fixed epoch count for sweep
            model.train()
            total_loss = 0
            
            for src, trg, src_lens, trg_lens in train_loader:
                src = src.to(device)
                trg = trg.to(device)
                
                optimizer.zero_grad()
                output = model(src, src_lens, trg)
                
                # Calculate loss
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                
                loss = criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                
                total_loss += loss.item()
            file_path = os.path.join(wandb.run.dir, "model.pth")
            torch.save(model.state_dict(), file_path)
            wandb.save('model.pth')
            # Validation
            #val_cer = evaluate_cer(model, dev_loader, device)
            #val_wer = evaluate_wer(model, dev_loader, device)
            val_acc = evaluate(model, dev_loader, device)
            wandb.log({
                'epoch': epoch,
                'train_loss': total_loss/len(train_loader),
                'val_acc': val_acc
            })

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src, trg, src_lens, trg_lens in loader:
            src = src.to(device)
            trg = trg.to(device)
            
            outputs = model(src, src_lens, trg, 0)  # No teacher forcing
            outputs = outputs.argmax(dim=-1)
            
            # Calculate accuracy
            mask = (trg != 0)
            correct += ((outputs == trg) * mask).sum().item()
            total += mask.sum().item()
    
    return correct / total

