# data preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext
import tqdm
import evaluate
import torchtext


seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

dataset = datasets.load_dataset("csv", data_files = "data_ass2/ko_en_parallel.csv")
train_valTest = dataset['train'].train_test_split(test_size=0.2)
val_test = train_valTest["test"].train_test_split(test_size=0.5)
dataset = {
    "train": train_valTest["train"],
    "validation": val_test["train"],
    "test": val_test["test"],
}

train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)


en_nlp = spacy.load("en_core_web_sm")
ko_nlp = spacy.load("ko_core_news_sm")


def tokenize_example(example, en_nlp, ko_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
    ko_tokens = [token.text for token in ko_nlp.tokenizer(example["ko"])][:max_length]
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        ko_tokens = [token.lower() for token in ko_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    ko_tokens = [sos_token] + ko_tokens + [eos_token]
    return {"en_tokens": en_tokens, "ko_tokens": ko_tokens}
    
max_length = 128
lower = True
sos_token = "<sos>"
eos_token = "<eos>"

fn_kwargs = {
    "en_nlp": en_nlp,
    "ko_nlp": ko_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)


from torchtext.vocab import build_vocab_from_iterator

min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

# Define your special tokens

# Initialize tokenizers for English and German (or your specific languages)

# Function to yield tokens from your data

# Assuming train_data["en_tokens"] and train_data["ko_tokens"] are lists of sentences
en_vocab = build_vocab_from_iterator(
    train_data["en_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
    special_first=True  # Place special tokens at the beginning of the vocab
)

ko_vocab = build_vocab_from_iterator(
    train_data["ko_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
    special_first=True  # Place special tokens at the beginning of the vocab
)

# Optional: Set default index for unknown tokens
en_vocab.set_default_index(en_vocab[unk_token])
ko_vocab.set_default_index(ko_vocab[unk_token])

fn_kwargs = {
    "en_vocab": en_vocab,
    "ko_vocab": ko_vocab,
}

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

en_vocab.set_default_index(unk_index)
ko_vocab.set_default_index(unk_index)

def numericalize_example(example, en_vocab, ko_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    ko_ids = ko_vocab.lookup_indices(example["ko_tokens"])
    return {"en_ids": en_ids, "ko_ids": ko_ids}

train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs, remove_columns=['en_tokens', 'ko_tokens'])
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs, remove_columns=['en_tokens', 'ko_tokens'])
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs, remove_columns=['en_tokens', 'ko_tokens'])
data_type = "torch"
format_columns = ["en_ids", "ko_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_ko_ids = [example["ko_ids"] for example in batch]
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_ko_ids = nn.utils.rnn.pad_sequence(batch_ko_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "ko_ids": batch_ko_ids,
        }
        return batch

    return collate_fn
    
def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

batch_size = 512
train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################
## transformer

# import torch
# import torch.nn as nn
# import torch.optim as optim

# import torchtext
# from torchtext.legacy.datasets import Multi30k
# from torchtext.legacy.data import Field, BucketIterator

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# import spacy
# import numpy as np

# import random
import math
import time

# SEED = 1234

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

# spacy_de = spacy.load('ko_core_news_sm')
# spacy_en = spacy.load('en_core_web_sm')


# def tokenize_de(text):
#     """
#     Tokenizes German text from a string into a list of strings
#     """
#     return [tok.text for tok in spacy_de.tokenizer(text)]

# def tokenize_en(text):
#     """
#     Tokenizes English text from a string into a list of strings
#     """
#     return [tok.text for tok in spacy_en.tokenizer(text)]

# SRC = Field(tokenize = tokenize_de, 
#             init_token = '<sos>', 
#             eos_token = '<eos>', 
#             lower = True, 
#             batch_first = True)

# TRG = Field(tokenize = tokenize_en, 
#             init_token = '<sos>', 
#             eos_token = '<eos>', 
#             lower = True, 
#             batch_first = True)
    
# train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
#                                                     fields = (SRC, TRG))

# SRC.build_vocab(train_data, min_freq = 2)
# TRG.build_vocab(train_data, min_freq = 2)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BATCH_SIZE = 128

# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data), 
#      batch_size = BATCH_SIZE,
#      device = device)


class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = max_length):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(input_dim, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]

        # Q = self.fc_q(query)

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2).to(device)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention

# INPUT_DIM = len(SRC.vocab)
# OUTPUT_DIM = len(TRG.vocab)
INPUT_DIM = len(ko_vocab)
OUTPUT_DIM = len(en_vocab)
HID_DIM = 512
ENC_LAYERS = 4
DEC_LAYERS = 4
ENC_HEADS = 4
DEC_HEADS = 4
ENC_PF_DIM = 1024
DEC_PF_DIM = 1024
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device).to(device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device).to(device)

# SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
# TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
SRC_PAD_IDX = pad_index # ko_vocab[pad_index]
TRG_PAD_IDX = pad_index # en_vocab[pad_index]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)

LEARNING_RATE = 0.0005


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in tqdm.tqdm(enumerate(iterator)):
        src = batch['ko_ids'].transpose(0,1).to(device)
        trg = batch['en_ids'].transpose(0,1).to(device)

        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch['ko_ids'].transpose(0,1).to(device)
            trg = batch['en_ids'].transpose(0,1).to(device)

            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 2
CLIP = 1

best_valid_loss = float('inf')

for epoch in tqdm.tqdm(range(N_EPOCHS)):
    
    start_time = time.time()
    
    # train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    # valid_loss = evaluate(model, valid_iterator, criterion)
    train_loss = train(model, train_data_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_data_loader, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load('tut6-model.pt'))

test_loss = evaluate(model, test_data_loader, criterion)


print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


# def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    
#     model.eval()
        
#     if isinstance(sentence, str):
#         tokens = [token.lower() for token in sentence]
#     else:
#         tokens = [token.lower() for token in sentence]

#     # tokens = [src_field.init_token] + tokens + [src_field.eos_token]
#     tokens = [sos_token] + tokens + [eos_token]
#     # src_indexes = [src_field.vocab.stoi[token] for token in tokens]
#     src_indexes = ko_vocab.lookup_indices(tokens)

#     src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
#     src_mask = model.make_src_mask(src_tensor)
    
#     with torch.no_grad():
#         enc_src = model.encoder(src_tensor, src_mask)

#     # trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
#     trg_indexes = en_vocab.lookup_indices([sos_token])

#     for i in range(max_len):

#         trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

#         trg_mask = model.make_trg_mask(trg_tensor)
        
#         with torch.no_grad():
#             output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
#         pred_token = output.argmax(2)[:,-1].item()
        
#         trg_indexes.append(pred_token)

#         # if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
#         if pred_token == eos_token:

#             break
    
#     # trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
#     trg_tokens = en_vocab.lookup_tokens(trg_indexes)
    
#     return trg_tokens[1:], attention


# example_idx = 8

# print(train_data)
# src = train_data['ko'][example_idx]
# trg = train_data['en'][example_idx]

# print(f'src = {src}')
# print(f'trg = {trg}')


# translation, attention = translate_sentence(src, ko_vocab, en_vocab, model, device)

# print(f'predicted trg = {translation}')

# from torchtext.data.metrics import bleu_score

# def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    
#     trgs = []
#     pred_trgs = []
    
#     for e, datum in tqdm.tqdm(enumerate(data)):
        
#         src = datum['ko']
#         trg = datum['en']
        
#         pred_trg, _ = translate_sentence(src, ko_vocab, en_vocab, model, device, max_len)
        
#         #cut off <eos> token
#         pred_trg = pred_trg[:-1]
        
#         pred_trgs.append(pred_trg)
#         trgs.append([trg])

#         if e > 100:
#             break

        
#     return bleu_score(pred_trgs, trgs)

# bleu_score = calculate_bleu(test_data, ko_vocab, en_vocab, model, device)

# print(f'BLEU score = {bleu_score*100:.2f}')
