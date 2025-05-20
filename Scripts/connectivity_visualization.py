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
import plotly.graph_objects as go
from IPython.display import HTML
import numpy as np


def get_connectivity_matrix(model, word, src_vocab, trg_vocab, device):
    original_mode = model.training
    model.train()  # Force training mode for gradient computation
    connectivity = np.zeros((len(word), len(word)))

    try:
        # Prepare input sequence
        src_indices = [src_vocab.char2idx['<go>']] + \
                     [src_vocab.char2idx.get(c, src_vocab.char2idx['<unk>']) for c in word] + \
                     [src_vocab.char2idx['<stop>']]
        src_tensor = torch.LongTensor(src_indices).unsqueeze(1).to(device)
        src_len = torch.LongTensor([len(src_indices)]).cpu()

        # Forward pass with gradient tracking
        with torch.enable_grad():
            # Get encoder outputs and explicitly retain gradients
            encoder_outputs, encoder_hidden = model.encoder(src_tensor.to(device), src_len.to(device))
            encoder_outputs.retain_grad()  # Critical fix: retain gradients for non-leaf tensor

            # Initialize decoder
            decoder_input = torch.LongTensor([[trg_vocab.char2idx['<go>']]]).to(device)
            decoder_hidden = encoder_hidden

            for t_step in range(len(word)):
                # Forward pass through decoder
                decoder_output, decoder_hidden, _ = model.decoder(
                    decoder_input.squeeze(0), decoder_hidden, encoder_outputs
                )
                
                # Get gradient from maximum logit
                target_idx = torch.argmax(decoder_output)
                logit_value = decoder_output[0, target_idx]
                
                # Backward pass
                model.zero_grad()
                logit_value.backward(retain_graph=True)

                # Extract gradients (now populated due to retain_grad())
                if encoder_outputs.grad is not None:
                    grad_importance = encoder_outputs.grad[1:-1].abs().sum(dim=2).squeeze().cpu().numpy()
                    valid_length = min(len(grad_importance), len(word))
                    connectivity[t_step, :valid_length] = grad_importance[:valid_length]
                
                # Update decoder input
                decoder_input = torch.LongTensor([[target_idx]]).to(device)

    finally:
        model.train(original_mode)  # Restore original mode

    return connectivity

def plot_connectivity(matrix, source_word, target_word, font_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    hindi_font = FontProperties(fname=font_path)
    
    # Plot matrix
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    
    # Set labels
    ax.set_xticks(np.arange(len(source_word)))
    ax.set_yticks(np.arange(len(target_word)))
    
    ax.set_xticklabels(list(source_word))
    ax.set_yticklabels(list(target_word), fontproperties=hindi_font)
    
    # Rotate and align
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Information Flow Strength", rotation=-90, va="bottom")
    
    plt.title("Transliteration Connectivity Pattern")
    plt.xlabel("Source Characters (Latin)")
    plt.ylabel("Target Characters (Devanagari)")
    plt.tight_layout()
    plt.show()

# Usage example
source_word = "namaste"
target_word = "नमस्ते"  # Model's prediction
font_path = "NotoSansDevanagari-Regular.ttf"

connectivity_matrix = get_connectivity_matrix(model, source_word, src_vocab, trg_vocab, device)
plot_connectivity(connectivity_matrix, source_word, target_word, font_path)

def create_interactive_connectivity(connectivity_matrix, source_word, target_word, font_path=None):
    # Prepare labels (source: Latin, target: Devanagari)
    source_chars = list(source_word)
    target_chars = list(target_word)
    
    # Create annotated heatmap
    fig = go.Figure(go.Heatmap(
        z=connectivity_matrix,
        x=source_chars,
        y=target_chars,
        colorscale='YlGnBu',
        hoverongaps=False,
        text=np.around(connectivity_matrix, decimals=3),
        texttemplate="%{text}",
        textfont={"size":12}
    ))
    
    # Add annotations and styling
    fig.update_layout(
        title=f'Transliteration Connectivity: {source_word} → {target_word}',
        xaxis_title='Source Characters (Latin)',
        yaxis_title='Target Characters (Devanagari)',
        font=dict(
            family=font_path if font_path else 'Noto Sans Devanagari, Arial',
            size=14
        ),
        width=800,
        height=600,
        hoverlabel=dict(
            bgcolor="white", 
            font_size=16,
            font_family="Rockwell"
        )
    )
    
    # Add custom hover template
    fig.update_traces(
        hovertemplate=(
            "<b>Source:</b> %{x}<br>"
            "<b>Target:</b> %{y}<br>"
            "<b>Connection Strength:</b> %{z:.3f}<extra></extra>"
        )
    )
    
    return fig

# Example usage
source_word = "namaste"
target_word = "नमस्ते"
connectivity_matrix = get_connectivity_matrix(model, source_word, src_vocab, trg_vocab, device)

# Create interactive plot
fig = create_interactive_connectivity(
    connectivity_matrix, 
    source_word, 
    target_word,
    font_path="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari&display=swap"
)

# Save as standalone HTML
fig.write_html("transliteration_connectivity.html")

# Display inline in notebook
HTML(fig.to_html())
