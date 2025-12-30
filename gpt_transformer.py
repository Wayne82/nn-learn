import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
block_size = 64

eval_iters = 200
eval_interval = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

torch.manual_seed(42)

# Define the simple GPT transformer model
class GPTTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=1, n_layer=1):
        super().__init__()

        # Define layers
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        # Forward pass
        B, T = x.shape

        token_embd = self.token_embedding(x) # (B, T, n_embd)
        position_embd = self.position_embedding(torch.arange(T, device=device)) # (T, n_embd)
        x = token_embd + position_embd #(B, T, n_embd)
        x = self.blocks(x) #(B, T, n_embd)
        x = self.ln_f(x) #(B, T, n_embd)
        logits = self.lm_head(x) #(B, T, vocab_size)

        return logits

    def print_params(self):
        print(sum(p.numel() for p in self.parameters())/1e6, 'M parameters')

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.sa(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        out = wei @ v # (B, T, head_size)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

# Define the trainer class
class Trainer():
    def __init__(self, data_loader, model, learning_rate=1e-3):
        self.data_loader = data_loader
        self.model = model.to(device)
        self.train_data = data_loader.train_data
        self.val_data = data_loader.val_data
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train(self, max_iters = 5000):
        # Train the model

        for iter in range(max_iters):
            # Evaluate loss on train and val sets
            if iter % eval_interval == 0:
                train_loss, val_loss = self.estimate_loss()
                print(f"Step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

            # Get batch of samples
            xb, yb = self.data_loader.get_batch('train')

            # Forward pass
            logits = self.model(xb)

            # Compute loss
            loss = self.compute_loss(logits, yb)

            # Backward pass and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_loss(self, logits, targets):
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return loss

    @torch.no_grad()
    def estimate_loss(self):
        # Estimate the loss
        self.model.eval()
        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = self.data_loader.get_batch(split)
                logits = self.model(xb)
                loss = self.compute_loss(logits, yb)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out['train'], out['val']

    @torch.no_grad()
    def generate(self, x, max_new_tokens):
        x = x.to(device)
        # Generate new tokens
        for _ in range(max_new_tokens):
            idx = x[:, -block_size:]
            logits = self.model(idx)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat((x, next_token), dim=1) # (B, T+1)
        return x

# Define the data loader class
class DataLoader():
    def __init__(self, data_path, batch_size=16):
        self.data_path = data_path
        self.batch_size = batch_size

    def load_data(self):
        # Load the data
        with open(self.data_path, 'r') as f:
            data = f.read()

        chars = sorted(list(set(data)))
        stoi = {ch:i for i, ch in enumerate(chars)}
        itos = {i:ch for i, ch in enumerate(chars)}

        self.vocab_size = len(chars)
        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: ''.join([itos[i] for i in l])
        self.data = torch.tensor(self.encode(data), dtype=torch.long)

        # Prepare the data
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split):
        # Get a batch of data
        _data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(0, len(_data) - block_size, (self.batch_size,))
        x = torch.stack([_data[i:i+block_size] for i in ix])
        y = torch.stack([_data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    def get_vocab_size(self):
        # Get the vocabulary size
        return self.vocab_size

    def print_samples(self, n=100):
        # Print sample data
        print(self.decode(self.val_data[:n].tolist()))