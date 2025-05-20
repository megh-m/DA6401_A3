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

# Dataset Configuration
DATASET_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
DATA_DIR = "./dakshina_dataset"
HI_LEXICON_DIR = os.path.join(DATA_DIR,"dakshina_dataset_v1.0", "hi", "lexicons") #For Hindi (Chosen Language)

def download_and_extract_dataset(): #Scripted Dataset Download
    if not os.path.exists(DATA_DIR):
        print("Downloading dataset...")
        response = requests.get(DATASET_URL)
        file = tarfile.open(fileobj=BytesIO(response.content))
        file.extractall(DATA_DIR)
        print("Dataset extracted successfully")

class TransliterationVocabulary: #Build Character Vocab and add go,stop, padding and unknown tokens
    def __init__(self):
        self.char2idx = defaultdict(lambda: len(self.char2idx))
        self.idx2char = {}
        self.special_tokens = ['<pad>', '<go>', '<stop>', '<unk>']
        
        # Initialize special tokens
        for token in self.special_tokens:
            self.char2idx[token]
        
        self.idx2char = {v: k for k, v in self.char2idx.items()}
    
    def add_word(self, word):
        #print(word) #for debugging
        for char in word:
            self.char2idx[char]
        self.idx2char = {v: k for k, v in self.char2idx.items()}

class TransliterationDataset(Dataset): #Dataset loader for Hindi
    def __init__(self, split='train'):
        self.split = split
        self.data = self._load_data()
        self.src_vocab = TransliterationVocabulary()
        self.trg_vocab = TransliterationVocabulary()
        
        # Build vocabularies
        for src,trg in self.data:
            self.src_vocab.add_word(src)
            self.trg_vocab.add_word(trg)
    
    def _load_data(self):
        """Load data from TSV files and filter non-string entries"""
        file_map = {
            'train': 'hi.translit.sampled.train.tsv',
            'dev': 'hi.translit.sampled.dev.tsv',
            'test': 'hi.translit.sampled.test.tsv'
        }
        
        df = pd.read_csv(
            os.path.join(HI_LEXICON_DIR, file_map[self.split]),
            sep='\t', 
            header=None,
            names=['devanagari', 'latin', 'count'],
            dtype={'latin': str, 'devanagari': str, 'count':int}  # Force string type
        )
        
        # Filter out non-string entries and empty strings
        valid_entries = [
            (latin, devanagari) 
            for latin, devanagari in zip(df['latin'], df['devanagari'])
            if (isinstance(latin, str) and 
                isinstance(devanagari, str) and
                len(latin) > 0 and 
                len(devanagari) > 0)
        ]
        
        return valid_entries

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, trg = self.data[idx]
        return (
            [self.src_vocab.char2idx['<go>']] + 
            [self.src_vocab.char2idx[c] for c in src if c not in ['<go>','<stop>','<pad>','<unk>']] +
            [self.src_vocab.char2idx['<stop>']],
            [self.trg_vocab.char2idx['<go>']] + 
            [self.trg_vocab.char2idx[c] for c in trg if c not in ['<go>','<stop>','<pad>','<unk>']] +
            [self.trg_vocab.char2idx['<stop>']]
        )

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    
    src_lens = torch.tensor([len(x) for x in src_batch])  # CPU tensor
    trg_lens = torch.tensor([len(x) for x in trg_batch])  # CPU tensor
    
    src_pad = pad_sequence(
        [torch.tensor(x) for x in src_batch],
        padding_value=0
    ).to(device)  # Move padded data to device
    
    trg_pad = pad_sequence(
        [torch.tensor(x) for x in trg_batch],
        padding_value=0
    ).to(device)  # Move padded data to device
    
    return src_pad, trg_pad, src_lens, trg_lens  # Lengths stay on CPU

def get_dataloaders(batch_size=64):
    """Create train, dev, test dataloaders"""
    download_and_extract_dataset()
    
    train_dataset = TransliterationDataset('train')
    dev_dataset = TransliterationDataset('dev')
    test_dataset = TransliterationDataset('test')
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, 
                  shuffle=True, collate_fn=collate_fn),
        DataLoader(dev_dataset, batch_size=batch_size, 
                 collate_fn=collate_fn),
        DataLoader(test_dataset, batch_size=batch_size,
                 collate_fn=collate_fn),
        train_dataset.src_vocab,
        train_dataset.trg_vocab
    )
train_loader, dev_loader, test_loader, src_vocab, trg_vocab = get_dataloaders()
print(f"Source vocab size: {len(src_vocab.char2idx)}")
print(f"Target vocab size: {len(trg_vocab.char2idx)}")
print(f"Training batches: {len(train_loader)}")