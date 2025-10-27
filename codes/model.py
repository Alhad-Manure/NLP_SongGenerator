"""
Transformer-based Hindi Song Lyrics Generator
Using PyTorch!
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import json
from typing import List, Tuple
import re

# ==================== TRANSFORMER MODEL ====================

class PositionalEncoding(nn.Module):
    """Add positional information to embeddings"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerLyricsModel(nn.Module):
    """
    Transformer model for lyrics generation
    Much better than LSTM for capturing long-range dependencies
    """
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.1, max_len=512):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU works better than ReLU
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None):
        # src shape: (batch_size, seq_len)
        
        # Embed and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through transformer
        output = self.transformer(src, src_mask)
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Generate mask to prevent attending to future tokens"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


# ==================== DATASET ====================

class LyricsDataset(Dataset):
    """Dataset for transformer training"""
    
    def __init__(self, sequences, max_len=128):
        self.sequences = sequences
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Pad or truncate
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [0] * (self.max_len - len(seq))
        
        # Input is sequence, target is shifted by 1
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        
        return input_seq, target_seq


# ==================== TOKENIZER ====================

class HindiTokenizer:
    """Simple word-level tokenizer for Hindi"""
    
    def __init__(self, language='hindi'):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.vocab_size = 4
        self.language = language
    
    def _preprocess_hindi(self, text):
        """
        Clean Hindi text while preserving structure
        Only removes non-Devanagari characters, keeps natural spacing
        """
        import unicodedata
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        # Remove only non-Devanagari and non-whitespace characters
        # This preserves line breaks and natural structure
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        # Remove excessive spaces but keep single newlines
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
        return text.strip()
    
    def fit(self, texts):
        """Build vocabulary from texts"""
        print(f"\nBuilding vocabulary from {len(texts)} songs...")
        
        for idx, text in enumerate(texts):
            if self.language == 'hindi':
                text = self._preprocess_hindi(text)
            
            # Split by whitespace (includes newlines)
            words = text.split()
            for word in words:
                if word and word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} songs... Vocab size: {self.vocab_size}")
        
        print(f"✓ Final vocabulary size: {self.vocab_size}")
        print(f"  Sample words: {list(self.word2idx.keys())[4:14]}")
    
    def encode(self, text, add_special_tokens=True):
        """Convert text to indices"""
        if self.language == 'hindi':
            text = self._preprocess_hindi(text)
        
        words = text.split()
        indices = []
        
        if add_special_tokens:
            indices.append(self.word2idx['<START>'])
        
        for word in words:
            if word:  # Skip empty strings
                indices.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        
        if add_special_tokens:
            indices.append(self.word2idx['<END>'])
        
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Convert indices back to text"""
        words = []
        special_tokens = {'<PAD>', '<UNK>', '<START>', '<END>'}
        
        for idx in indices:
            if idx >= self.vocab_size:
                continue
            word = self.idx2word[idx]
            if skip_special_tokens and word in special_tokens:
                continue
            words.append(word)
        
        return ' '.join(words)


# ==================== TRAINING ====================

class TransformerLyricsTrainer:
    """Trainer for transformer model"""
    
    def __init__(self, config_file='config/transformer_config.json'):
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = HindiTokenizer(language=self.config.get('language', 'hindi'))
        self.model = None
    
    def prepare_data(self, lyrics_file, song_separator='\n--\n'):
        """Load and prepare data"""
        print(f"\nLoading lyrics from: {lyrics_file}")
        
        with open(lyrics_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by song separator
        songs = content.split(song_separator)
        songs = [song.strip() for song in songs if song.strip()]
        
        print(f"Found {len(songs)} songs")
        
        # Build vocabulary
        self.tokenizer.fit(songs)
        
        # Create sequences
        sequences = []
        for song in songs:
            seq = self.tokenizer.encode(song)
            sequences.append(seq)
        
        return sequences
    
    def train(self, sequences):
        """Train the model"""
        print(f"\n{'='*60}")
        print("TRAINING TRANSFORMER MODEL")
        print(f"{'='*60}\n")
        
        # Create dataset and dataloader
        dataset = LyricsDataset(sequences, max_len=self.config['max_seq_length'])
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        # Initialize model
        self.model = TransformerLyricsModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            max_len=self.config['max_seq_length']
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training loop
        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Generate mask
                seq_len = input_seq.size(1)
                mask = self.model.generate_square_subsequent_mask(seq_len).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output = self.model(input_seq, mask)
                
                # Calculate loss
                loss = criterion(
                    output.reshape(-1, self.tokenizer.vocab_size),
                    target_seq.reshape(-1)
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.config['epochs']}] "
                          f"Batch [{batch_idx+1}/{len(dataloader)}] "
                          f"Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step()
            
            print(f"\nEpoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f"checkpoints/checkpoint_epoch_{epoch+1}.pt")
        
        print("✓ Training complete!")
    
    def generate(self, seed_text="मेरा", max_length=100, temperature=0.9, top_k=50):
        """Generate lyrics"""
        self.model.eval()
        
        # Encode seed text
        tokens = self.tokenizer.encode(seed_text, add_special_tokens=True)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                input_seq = torch.tensor([tokens], dtype=torch.long).to(self.device)
                
                # Generate mask
                seq_len = input_seq.size(1)
                mask = self.model.generate_square_subsequent_mask(seq_len).to(self.device)
                
                # Get predictions
                output = self.model(input_seq, mask)
                logits = output[0, -1, :] / temperature
                
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices[torch.multinomial(probs, 1)]
                
                # Add to sequence
                tokens.append(next_token.item())
                
                # Stop if end token
                if next_token.item() == self.tokenizer.word2idx['<END>']:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(tokens)
        return generated_text
    
    def save_model(self, path):
        """Save model and tokenizer"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model and tokenizer"""
        checkpoint = torch.load(path, map_location=self.device)
        self.tokenizer = checkpoint['tokenizer']
        self.config = checkpoint['config']
        
        self.model = TransformerLyricsModel(
            vocab_size=self.tokenizer.vocab_size,
            **{k: v for k, v in self.config.items() if k in [
                'd_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout'
            ]}
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


# ==================== USAGE ====================

if __name__ == "__main__":

    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./config", exist_ok=True)

    # Create config
    config = {
        "language": "hindi",
        "max_seq_length": 128,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "batch_size": 8,
        "learning_rate": 0.0001,
        "epochs": 50
    }
    
    # Save config
    with open('config/transformer_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer
    trainer = TransformerLyricsTrainer('config/transformer_config.json')
    
    # Prepare data
    sequences = trainer.prepare_data('lyrics_files/all_hindi_combined.txt', song_separator='\n--\n')
    
    # Train
    trainer.train(sequences)
    
    # Generate lyrics
    print(f"\n{'='*60}")
    print("GENERATING LYRICS")
    print(f"{'='*60}\n")
    
    for i in range(3):
        lyrics = trainer.generate(
            seed_text="तेरी यादों में",
            max_length=100,
            temperature=0.9,
            top_k=50
        )
        print(f"\n--- Generated Lyrics {i+1} ---")
        print(lyrics)
        print("-" * 60)
    
    # Save final model
    trainer.save_model('checkpoints/transformer_hindi_lyrics.pt')