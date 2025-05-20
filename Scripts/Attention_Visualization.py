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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import urllib.request
import os
import torch

def get_model_attention_data(model, data_loader, src_vocab, trg_vocab, device, num_examples=9):
    """Extract attention matrices and tokens from the model for visualization
    
    Args:
        model: The seq2seq model with attention
        data_loader: DataLoader with test/validation examples
        src_vocab, trg_vocab: Source and target vocabularies
        device: Device to run inference on
        num_examples: Number of examples to process
        
    Returns:
        attentions: List of attention matrices
        src_tokens: List of source token sequences
        trg_tokens: List of target token sequences
    """
    model.eval()
    attentions = []
    src_tokens = []
    trg_tokens = []
    example_count = 0
    
    with torch.no_grad():
        for src, trg, src_lens, trg_lens in data_loader:
            src = src.to(device)
            
            # Process batch items one by one to collect attention
            for i in range(src.size(1)):  # Iterate through batch
                if example_count >= num_examples:
                    break
                
                # Get source sequence
                src_seq = src[:, i:i+1]  # Keep dimensions
                src_len = src_lens[i:i+1]
                
                # Initialize decoder input
                decoder_input = torch.LongTensor([[trg_vocab.char2idx['<go>']]]).to(device)
                
                # Encode source
                encoder_outputs, encoder_hidden = model.encoder(src_seq, src_len)
                decoder_hidden = encoder_hidden
                
                # Initialize containers
                step_attentions = []
                pred_indices = []
                
                # Greedy decoding with attention collection
                for _ in range(50):  # Max length limit
                    # Forward pass through decoder with attention
                    decoder_output, decoder_hidden, attn_weights = model.decoder(
                        decoder_input.squeeze(0), decoder_hidden, encoder_outputs
                    )
                    
                    # Store attention weights
                    step_attentions.append(attn_weights.squeeze().cpu().numpy())
                    
                    # Get most likely next token
                    topi = decoder_output.argmax(1)
                    pred_indices.append(topi.item())
                    
                    # Break if stop
                    if topi.item() == trg_vocab.char2idx['<stop>']:
                        break
                    
                    # Next input is current prediction
                    decoder_input = topi.unsqueeze(0)
                
                # Convert source tokens to characters (excluding special tokens)
                src_indices = src_seq.squeeze().cpu().numpy()
                src_chars = [src_vocab.idx2char[idx] for idx in src_indices 
                           if idx not in [0,3]]
                
                # Convert predictions to characters (excluding special tokens)
                trg_chars = [trg_vocab.idx2char[idx] for idx in pred_indices 
                           if idx not in [0,3]]
                
                # Extract attention matrix (target_len x source_len)
                attn_matrix = np.array(step_attentions)
                
                # Trim attention to actual sequence lengths
                attn_matrix = attn_matrix[:len(trg_chars), :len(src_chars)]
                
                # Store results
                attentions.append(attn_matrix)
                src_tokens.append(src_chars)
                trg_tokens.append(trg_chars)
                
                example_count += 1
                if example_count >= num_examples:
                    break
            
            if example_count >= num_examples:
                break
    #df = pd.DataFrame([attentions, src_tokens, trg_tokens])
    #print(df)
    return attentions, src_tokens, trg_tokens

# Download Noto Sans Devanagari font if not present
font_url = 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf'
font_path = 'NotoSansDevanagari-Regular.ttf'
if not os.path.exists(font_path):
    urllib.request.urlretrieve(font_url, font_path)

hindi_font = FontProperties(fname=font_path)

# Load model and data
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

model = best_model
model.to(device)
model.eval()

# Get real attention data
_, _, test_loader, src_vocab, trg_vocab = get_dataloaders(batch_size=9)
attentions, src_tokens, trg_tokens = get_model_attention_data(
    model, test_loader, src_vocab, trg_vocab, device, num_examples=9
)

# Plot 3x3 grid of attention heatmaps
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    if i >= len(attentions):
        ax.axis('off')
        continue
        
    attn = attentions[i]
    src = src_tokens[i]
    trg = trg_tokens[i]
    
    # Create heatmap
    im = ax.matshow(attn, cmap='viridis')
    ax.set_xticks(np.arange(len(src)))
    ax.set_yticks(np.arange(len(trg)))
    ax.set_xticklabels(src, fontsize=10)
    ax.set_yticklabels(trg, fontproperties=hindi_font, fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
    
    # Add source and target word labels
    src_word = ''.join(src)
    trg_word = ''.join(trg)
    #ax.set_title(f'{src_word} → {trg_word}', fontproperties=hindi_font, fontsize=14)

# Add colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label('Attention Weight', rotation=270, labelpad=15)

#plt.tight_layout()
plt.savefig('attention_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
