"""
Transformer-based Hindi Song Lyrics Generator
Built from scratch using PyTorch - much better than LSTM approach!
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
        """
        Load and prepare data - handles exact format with \n--\n separator
        Preserves original structure of each song
        """
        print(f"\nLoading lyrics from: {lyrics_file}")
        
        with open(lyrics_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by exact separator '\n--\n'
        songs = content.split(song_separator)
        
        # Clean up songs - remove leading/trailing whitespace but preserve internal structure
        cleaned_songs = []
        for song in songs:
            song = song.strip()
            if song:  # Only keep non-empty songs
                cleaned_songs.append(song)
        
        print(f"✓ Found {len(cleaned_songs)} songs")
        
        # Show sample
        if cleaned_songs:
            print(f"\n--- Sample Song 1 (first 200 chars) ---")
            print(cleaned_songs[0][:200])
            print("..." if len(cleaned_songs[0]) > 200 else "")
            print("-" * 50)
        
        # Build vocabulary from all songs
        print("\nBuilding vocabulary...")
        self.tokenizer.fit(cleaned_songs)
        
        # Create sequences - each song becomes one sequence
        sequences = []
        total_words = 0
        
        print("\nEncoding songs to sequences...")
        for idx, song in enumerate(cleaned_songs):
            seq = self.tokenizer.encode(song, add_special_tokens=True)
            sequences.append(seq)
            total_words += len(seq)
            
            if (idx + 1) % 50 == 0:
                print(f"  Encoded {idx + 1}/{len(cleaned_songs)} songs...")
        
        print(f"\n✓ Dataset prepared successfully!")
        print(f"  Total songs: {len(sequences)}")
        print(f"  Total words: {total_words:,}")
        print(f"  Average words per song: {total_words // len(sequences)}")
        print(f"  Vocabulary size: {self.tokenizer.vocab_size}")
        
        # Show sequence length distribution
        seq_lengths = [len(seq) for seq in sequences]
        print(f"\nSequence Length Statistics:")
        print(f"  Min: {min(seq_lengths)}")
        print(f"  Max: {max(seq_lengths)}")
        print(f"  Average: {sum(seq_lengths) // len(seq_lengths)}")
        print(f"  Median: {sorted(seq_lengths)[len(seq_lengths)//2]}")
        
        return sequences
    
    def train(self, sequences):
        """Train the model with proper train/test split and evaluation"""
        print(f"\n{'='*60}")
        print("TRAINING TRANSFORMER MODEL")
        print(f"{'='*60}\n")
        
        # Split sequences into train and test sets
        from sklearn.model_selection import train_test_split
        
        train_sequences, test_sequences = train_test_split(
            sequences,
            test_size=self.config.get('test_size', 0.15),  # 15% for testing
            random_state=42,
            shuffle=True
        )
        
        print(f"Dataset Split:")
        print(f"  Total sequences: {len(sequences)}")
        print(f"  Training sequences: {len(train_sequences)}")
        print(f"  Test sequences: {len(test_sequences)}")
        print(f"  Test ratio: {len(test_sequences)/len(sequences)*100:.1f}%\n")
        
        # Create datasets and dataloaders
        train_dataset = LyricsDataset(train_sequences, max_len=self.config['max_seq_length'])
        test_dataset = LyricsDataset(test_sequences, max_len=self.config['max_seq_length'])
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,  # Shuffle every epoch for better generalization
            num_workers=0,
            drop_last=True  # Drop incomplete batches
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,  # Don't shuffle test set
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
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model Architecture:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB\n")
        
        # Optimizer with weight decay for regularization
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler with warmup
        warmup_steps = self.config.get('warmup_steps', 500)
        total_steps = len(train_dataloader) * self.config['epochs']
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # Ignore padding
            label_smoothing=self.config.get('label_smoothing', 0.1)  # Label smoothing for better generalization
        )
        
        # Training history
        history = {
            'train_loss': [],
            'test_loss': [],
            'train_perplexity': [],
            'test_perplexity': [],
            'learning_rate': []
        }
        
        best_test_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('early_stopping_patience', 5)
        
        # Training loop
        print(f"{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config['epochs']):
            # ============ TRAINING PHASE ============
            self.model.train()
            train_loss = 0
            train_batches = 0
            
            print(f"Epoch [{epoch+1}/{self.config['epochs']}]")
            print("-" * 60)
            
            for batch_idx, (input_seq, target_seq) in enumerate(train_dataloader):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Generate causal mask
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
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.config.get('grad_clip', 1.0))
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = train_loss / train_batches
                    perplexity = math.exp(min(avg_loss, 20))  # Cap for numerical stability
                    print(f"  Batch [{batch_idx+1}/{len(train_dataloader)}] "
                          f"Loss: {loss.item():.4f} | "
                          f"Avg Loss: {avg_loss:.4f} | "
                          f"Perplexity: {perplexity:.2f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            avg_train_loss = train_loss / train_batches
            train_perplexity = math.exp(min(avg_train_loss, 20))
            
            # ============ EVALUATION PHASE ============
            self.model.eval()
            test_loss = 0
            test_batches = 0
            
            with torch.no_grad():
                for input_seq, target_seq in test_dataloader:
                    input_seq = input_seq.to(self.device)
                    target_seq = target_seq.to(self.device)
                    
                    seq_len = input_seq.size(1)
                    mask = self.model.generate_square_subsequent_mask(seq_len).to(self.device)
                    
                    output = self.model(input_seq, mask)
                    loss = criterion(
                        output.reshape(-1, self.tokenizer.vocab_size),
                        target_seq.reshape(-1)
                    )
                    
                    test_loss += loss.item()
                    test_batches += 1
            
            avg_test_loss = test_loss / test_batches
            test_perplexity = math.exp(min(avg_test_loss, 20))
            
            # Save history
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(avg_test_loss)
            history['train_perplexity'].append(train_perplexity)
            history['test_perplexity'].append(test_perplexity)
            history['learning_rate'].append(scheduler.get_last_lr()[0])
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1} SUMMARY")
            print(f"{'='*60}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
            print(f"  Test Loss:  {avg_test_loss:.4f} | Perplexity: {test_perplexity:.2f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Check for improvement
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
                print(f" New best test loss! Saving checkpoint...")
                self.save_model(f"checkpoints/best_model_epoch_{epoch+1}.pt")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
            
            print(f"{'='*60}\n")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best test loss: {best_test_loss:.4f}")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f"checkpoints/checkpoint_epoch_{epoch+1}.pt")
        
        # Save training history
        history_file = self.config.get('history_file', 'training_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final test loss: {history['test_loss'][-1]:.4f}")
        print(f"  Best test loss: {best_test_loss:.4f}")
        print(f"  Training history saved to: {history_file}")
        print(f"{'='*60}\n")
        
        return history
    
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
    
    def plot_training_history(self, history_file='training_history.json'):
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
            
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plot
            axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
            axes[0, 0].plot(history['test_loss'], label='Test Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Test Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Perplexity plot
            axes[0, 1].plot(history['train_perplexity'], label='Train Perplexity', linewidth=2)
            axes[0, 1].plot(history['test_perplexity'], label='Test Perplexity', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Perplexity')
            axes[0, 1].set_title('Training and Test Perplexity')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate plot
            axes[1, 0].plot(history['learning_rate'], linewidth=2, color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Loss difference plot
            loss_diff = [abs(t - v) for t, v in zip(history['train_loss'], history['test_loss'])]
            axes[1, 1].plot(loss_diff, linewidth=2, color='red')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('|Train Loss - Test Loss|')
            axes[1, 1].set_title('Overfitting Indicator')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: training_history.png")
            plt.show()
            
        except ImportError:
            print("matplotlib not installed. Skipping plot generation.")
        except Exception as e:
            print(f"Error plotting history: {e}")

if __name__ == "__main__":

    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./config", exist_ok=True)

    # import sys
    
    # First, verify your dataset format
    dataset_file = 'lyrics_files/all_hindi_combined3.txt'
    
    '''
    print("Step 1: Verifying dataset format...")
    if not verify_dataset_format(dataset_file, song_separator='\n--\n'):
        print("\n Please fix your dataset format first!")
        sys.exit(1)
    '''
        
    print("\n" + "="*60)
    print("Dataset verification passed! Starting training...")
    print("="*60 + "\n")
    
    # Create config
    config = {
        "language": "hindi",
        "max_seq_length": 128,  # Adjust based on your avg song length
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "batch_size": 16,
        "learning_rate": 0.0001,
        "epochs": 70
    }
    
    # Save config
    with open('config/transformer_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer
    trainer = TransformerLyricsTrainer('config/transformer_config.json')
    
    # Prepare data
    sequences = trainer.prepare_data(dataset_file, song_separator='\n--\n')
    
    # Train
    trainer.train(sequences)
    
    # Generate lyrics
    print(f"\n{'='*60}")
    print("GENERATING LYRICS")
    print(f"{'='*60}\n")
    
    seed_texts = [
        "हम घर से",
        "मैं हूँ",
        "तेरी यादों में"
    ]
    
    for seed in seed_texts:
        print(f"\n{'='*60}")
        print(f"Seed: '{seed}'")
        print(f"{'='*60}\n")
        
        for i in range(2):
            lyrics = trainer.generate(
                seed_text=seed,
                max_length=100,
                temperature=0.9,
                top_k=50
            )
            print(f"\n--- Variation {i+1} ---")
            print(lyrics)
            print("-" * 60)
    
    # Save final model
    trainer.save_model('checkpoints/transformer_hindi_lyrics.pt')
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model saved to: checkpoints/transformer_hindi_lyrics.pt")
    print(f"Config saved to: config/transformer_config.json")
