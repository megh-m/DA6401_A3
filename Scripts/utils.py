import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class CharEmbed(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(CharEmbed, self).__init__()
        self.embed = nn.Embedding(input_dim, embed_dim)
    
    def forward(self, input_seq):
        return self.embed(input_seq)

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers=1, 
                 cell_type='GRU', dropout=0.0, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.embed = nn.Embedding(input_dim, embed_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cell_type = cell_type
        self.bidirectional = bidirectional #to allow forward and backward time step data processing
        # Cell type options GRU, LSTM & vanilla RNN
        if cell_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, bidirectional=bidirectional)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, bidirectional=bidirectional)
        else: 
            self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, bidirectional=bidirectional)
    
    def forward(self, input_seq, input_lengths, hidden=None):
        # Sort sequences by length
        input_lengths, sort_idx = torch.sort(input_lengths, descending=True)
        input_seq = input_seq[:, sort_idx]  # (seq_len, batch_size, ...)
        
        # Convert to embeddings
        embedded = self.embed(input_seq)
        
        # Pack with enforce_sorted=False
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            input_lengths.cpu(), 
            enforce_sorted=False
        )
        
        # Forward pass
        outputs, hidden = self.rnn(packed, hidden)
        
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        
        # Restore original order
        _, unsort_idx = torch.sort(sort_idx)
        outputs = outputs[:, unsort_idx]
        
        # Handle LSTM hidden/cell states
        if isinstance(hidden, tuple):
            hidden = (
                hidden[0][:, unsort_idx],  # Hidden state
                hidden[1][:, unsort_idx]   # Cell state
            )
        else:  # For GRU/RNN
            hidden = hidden[:, unsort_idx]
        
        return outputs, hidden

class DecoderRNN(nn.Module): #Basically similar to the encoder, will have a softmax to predict next char
    def __init__(self, output_dim, embed_dim, hidden_dim, vocab, n_layers=1, cell_type='GRU', dropout=0.0, go_idx=1, stop_idx=2):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(output_dim, embed_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.cell_type = cell_type
        self.go_idx = go_idx
        self.stop_idx = stop_idx
        self.vocab = vocab
        if cell_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0)
        
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        # Get embedding of current input character
        embedded = self.embed(input).unsqueeze(0)
        
        # Forward pass through decoder
        output, hidden = self.rnn(embedded, hidden)
        
        # Predict next character probabilities
        output = self.softmax(self.out(output.squeeze(0)))
        
        return output, hidden

class Attention2(nn.Module):
    def __init__(self, hidden_size):
        super(Attention2, self).__init__()
        self.encoder_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder_proj = nn.Linear(hidden_size, hidden_size)
        self.energy = nn.Linear(hidden_size, 1)
        
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (num_layers, batch_size, hidden_size)
        # encoder_outputs: (seq_len, batch_size, hidden_size)
        
        # Project decoder hidden state
        decoder_projected = self.decoder_proj(decoder_hidden[-1].unsqueeze(1))  # (batch_size, 1, hidden_size)
        #decoder_projected = decoder_projected.permute(1,0,2)
        #print('decoder:',decoder_projected.size())
        # Project encoder outputs
        encoder_projected = self.encoder_proj(encoder_outputs.permute(1,0,2))  # (batch_size, seq_len, hidden_size)
        #print('encoder:',encoder_projected.size())
        # Calculate attention scores
        scores = self.energy(torch.tanh(decoder_projected + encoder_projected))  # (batch_size, seq_len, 1)
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Calculate context vector
        context = torch.bmm(attn_weights.permute(0,2,1), encoder_projected)  # (batch_size, 1, hidden_size)
        
        return context.squeeze(1), attn_weights.squeeze(2)  # (batch_size, hidden_size), (batch_size, seq_len)

class AttnDecoderRNN(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, vocab, n_layers=1, cell_type='GRU', dropout=0.0, go_idx = 1, stop_idx = 2):
        super(AttnDecoderRNN, self).__init__()
        self.embed = nn.Embedding(output_dim, embed_dim)
        self.attention = Attention2(hidden_dim)
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.go_idx = go_idx
        self.stop_idx = stop_idx
        self.vocab = vocab
        
        # RNN Cell Selection
        if cell_type == 'GRU':
            self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, 
                             n_layers, dropout=dropout if n_layers > 1 else 0)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim + hidden_dim, hidden_dim,
                              n_layers, dropout=dropout if n_layers > 1 else 0)
        else:
            self.rnn = nn.RNN(embed_dim + hidden_dim, hidden_dim,
                             n_layers, dropout=dropout if n_layers > 1 else 0)
            
        self.out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input: (batch_size)
        # hidden: (num_layers, batch_size, hidden_dim)
        # encoder_outputs: (seq_len, batch_size, hidden_dim)
        
        embedded = self.dropout(self.embed(input)).unsqueeze(0)  # (1, batch_size, emb_dim)
        
        # Calculate attention context
        context, attn_weights = self.attention(hidden, encoder_outputs)
        context = context.unsqueeze(0)  # (1, batch_size, hidden_dim)
        
        # Combine embedded input and context
        rnn_input = torch.cat((embedded, context), dim=2)  # (1, batch_size, emb_dim + hidden_dim)
        
        # RNN forward pass
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Final prediction
        output = self.out(torch.cat((output.squeeze(0), context.squeeze(0)), dim=1))
        output = F.log_softmax(output, dim=1)
        
        return output, hidden, attn_weights

class AttnSeq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, hidden_dim, num_layers, cell_type, dropout, device,vocab, go_idx=1, stop_idx=2):
        super(AttnSeq2Seq, self).__init__()
        self.go_idx = go_idx
        self.stop_idx = stop_idx
        self.vocab = vocab
        self.encoder = EncoderRNN(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=num_layers,
            cell_type=cell_type,
            dropout=dropout
        )

        # Internal decoder creation
        self.decoder = AttnDecoderRNN(
            output_dim=output_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=num_layers,
            cell_type=cell_type,
            dropout=dropout,
            vocab = vocab,
            go_idx = go_idx,
            stop_idx = stop_idx
        )
        self.device = device 
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.out.out_features
        
        # Encoder forward pass
        encoder_outputs, hidden = self.encoder(src, src_len)
        
        # Decoder initial input
        dec_input = trg[0,:]  # <go> token
        
        # Output tensor initialization
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        attentions = torch.zeros(trg_len, batch_size, src.shape[0]).to(self.device)
        
        for t in range(1, trg_len):
            dec_output, hidden, attn_weights = self.decoder(
                dec_input, hidden, encoder_outputs
            )
            
            outputs[t] = dec_output
            attentions[t] = attn_weights
            
            # Teacher forcing decision
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output.argmax(1)
            dec_input = trg[t] if teacher_force else top1
            
        return outputs, attentions
