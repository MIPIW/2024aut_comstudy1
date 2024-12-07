# data preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext
from tqdm import tqdm
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

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from collections import Counter
# import random
# import numpy as np
# import spacy
# import datasets
# from tqdm import tqdm
# import evaluate
# import nltk
# import re

# seed = 1234

# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# ## 구현,실험 전체적인 설명 및 분석
# ## Data
# # 1. 한-영 병렬 데이터 생성 후 학습 및 테스트 데이터 생성 (전처리 과정 포함)
# ## 1.1 파일 불러오기 및 자료 형식 확인
# # ko-en-en.parse.syn
# # 330,974 한국어 문장에 대응되는 영어문장이 '품사와 구문분석'이 되어 있는 파일
# en_file_path = './data_ass2/ko-en.en.parse.syn'
# # ko-en-ko.parse.syn
# # 이에 대응되는 한국어 문장이 '형태소와 구문분석'이 되어 있는 파일
# ko_file_path = './data_ass2/ko-en.ko.parse'

# # en_file_path
# with open(en_file_path, 'r', encoding='utf-8') as file:
#     for i in range(20):
#         line = file.readline().strip()
# # ko_file_path
# with open(ko_file_path, 'r', encoding='utf-8') as file:
#     for i in range(20):
#         line = file.readline().strip()
# ## 1.2 각 파일 처리하는 함수 정의
# from datasets import Dataset, DatasetDict
# import nltk
# # 한국어 파일을 처리하는 함수
# def get_sents_from_ko_parse(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         sentences = []
#         for line in f:
#             if line.startswith('<sent'):
#                 sentence = []
#             elif line.startswith('</sent>'):
#                 sentences.append(" ".join(sentence))
#             else:
#                 elems = line.split('\t')
#                 if len(elems) > 1:
#                     words = elems[-1].split("|")
#                     for word_with_label in words:
#                         sentence.append(word_with_label.split("/")[0])
#     return sentences

# # 영어 파일을 처리하는 함수(nltk 사용)
# def get_sents_from_en_parse(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return [" ".join(nltk.Tree.fromstring(line.strip()).leaves()) for line in f if line.strip()]
# # 데이터 불러오기
# ko_sents = get_sents_from_ko_parse(ko_file_path)
# en_sents = get_sents_from_en_parse(en_file_path)
# # 두 리스트의 길이를 가장 짧은 쪽에 맞춤
# min_size = min(len(ko_sents), len(en_sents))
# ko_sents = ko_sents[:min_size]
# en_sents = en_sents[:min_size]

# # 데이터 분할
# train_size = int(min_size * 0.8)
# valid_size = int(min_size * 0.1)
# test_size = min_size - (train_size + valid_size)
# # DatasetDict 생성
# dataset = DatasetDict({
#     "train": Dataset.from_dict({"ko": ko_sents[:train_size], "en": en_sents[:train_size]}),
#     "validation": Dataset.from_dict({"ko": ko_sents[train_size:train_size + valid_size], "en": en_sents[train_size:train_size + valid_size]}),
#     "test": Dataset.from_dict({"ko": ko_sents[train_size + valid_size:], "en": en_sents[train_size + valid_size:]})
# })
# train_data, valid_data, test_data = (
#     dataset["train"],
#     dataset["validation"],
#     dataset["test"],
# )
# ## 1.3 Tokenizer, Padding, Truncation
# def tokenize_example(example, max_length, lower, sos_token, eos_token):

#     # truncation
#     ko_tokens = example["ko"].split(" ")[:max_length]
#     en_tokens = example["en"].split(" ")[:max_length]

#     # 소문자화
#     if lower:
#         en_tokens = [token.lower() for token in en_tokens]

#     # special token 추가
#     ko_tokens = [sos_token] + ko_tokens + [eos_token]
#     en_tokens = [sos_token] + en_tokens + [eos_token]

#     return {"ko_tokens": ko_tokens, "en_tokens": en_tokens}
# max_length = 1_000
# lower = True
# sos_token = "<sos>"
# eos_token = "<eos>"

# fn_kwargs = {
#     "max_length": max_length,
#     "lower": lower,
#     "sos_token": sos_token,
#     "eos_token": eos_token,
# }

# # Dataset.map(function): function을 Dataset에 일괄 적용
# train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
# valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
# test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)
# train_data[15]
# ## 1.4 Torchtext 대체하는 class

# # torchtext를 대체하는 class
# # torchtext에 있던 build_vocab_from_iterator 대신할 class 정의
# from collections import Counter
# from tqdm import tqdm

# class Vocab:
#     def __init__(self):
#         self.stoi = {}  # string-to-index dictionary
#         self.itos = []  # index-to-string list

#     def build_vocab_from_iterator(self, iterator, min_freq=1, specials=["<unk>", "<pad>", "<sos>", "<eos>"]):
#         # 빈도수 계산
#         counter = Counter(token for tokens in iterator for token in tokens)
#         sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

#         # special tokens 추가
#         for i, special in enumerate(specials):
#             self.stoi[special] = i
#             self.itos.append(special)

#         # 빈도 조건에 맞는 토큰 추가
#         for token, freq in tqdm(sorted_tokens, desc="Building Vocab"):
#             if token not in specials and freq >= min_freq:
#                 index = len(self.itos)
#                 self.stoi[token] = index
#                 self.itos.append(token)

#     def __len__(self):
#         return len(self.itos)

#     def __contains__(self, token):
#         return token in self.stoi

#     def __getitem__(self, token):
#         return self.stoi.get(token, self.stoi["<unk>"])

#     def get_stoi(self):
#         return self.stoi

#     def get_itos(self):
#         return self.itos

#     def lookup_indices(self, tokens):
#         return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

#     def lookup_tokens(self, indices):
#         return [self.itos[index] if index < len(self.itos) else "<unk>" for index in indices]

#     def set_default_index(self, index):
#         self.stoi["<unk>"] = index


# min_freq = 2
# unk_token = "<unk>"
# pad_token = "<pad>"
# sos_token = "<sos>"
# eos_token = "<eos>"

# special_tokens = [
#     unk_token,
#     pad_token,
#     sos_token,
#     eos_token,
# ]

# # Vocab 클래스 인스턴스 생성
# ko_vocab = Vocab()
# en_vocab = Vocab()

# # 어휘 사전 구축 (train_data의 'ko_tokens'와 'en_tokens'를 사용)
# ko_vocab.build_vocab_from_iterator(
#     (tokens for tokens in train_data["ko_tokens"]),  # 한국어 토큰 리스트
#     min_freq=min_freq,
#     specials=special_tokens
# )

# en_vocab.build_vocab_from_iterator(
#     (tokens for tokens in train_data["en_tokens"]),  # 영어 토큰 리스트
#     min_freq=min_freq,
#     specials=special_tokens
# )
# assert en_vocab[unk_token] == ko_vocab[unk_token]
# assert en_vocab[pad_token] == ko_vocab[pad_token]

# unk_index = en_vocab[unk_token]
# pad_index = en_vocab[pad_token]

# # OOV는 '<unk>' 토큰에 대응됨
# ko_vocab.set_default_index(unk_index)
# en_vocab.set_default_index(unk_index)
# def numericalize_example(example, ko_vocab, en_vocab):
#     ko_ids = ko_vocab.lookup_indices(example["ko_tokens"])
#     en_ids = en_vocab.lookup_indices(example["en_tokens"])
#     return {"ko_ids": ko_ids, "en_ids": en_ids}
# # 각 데이터의 token에 해당하는 id 리스트 형성
# fn_kwargs = {"ko_vocab": ko_vocab, "en_vocab": en_vocab}

# train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
# valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
# test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)
# train_data[0]
# data_type = "torch"
# format_columns = ["ko_ids", "en_ids"]

# # Dataset.with_format(type, columns): Dataset의 columns 안의 데이터를 type 형태로 변경
# train_data = train_data.with_format(
#     type=data_type,
#     columns=format_columns,
#     output_all_columns=True
# )

# valid_data = valid_data.with_format(
#     type=data_type,
#     columns=format_columns,
#     output_all_columns=True,
# )

# test_data = test_data.with_format(
#     type=data_type,
#     columns=format_columns,
#     output_all_columns=True,
# )
# type(train_data[0]["ko_ids"])
# # Collate 함수 정의

# def get_collate_fn(pad_index):
#     def collate_fn(batch):
#         batch_ko_ids = [example["ko_ids"] for example in batch]
#         batch_en_ids = [example["en_ids"] for example in batch]
#         # 배치 단위 처리를 위래 길이 맞추기 위한 padding
#         batch_ko_ids = nn.utils.rnn.pad_sequence(batch_ko_ids, padding_value=pad_index)
#         batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
#         batch = {
#             "ko_ids": batch_ko_ids,
#             "en_ids": batch_en_ids,
#         }
#         return batch

#     return collate_fn
# # Dataloader 함수 정의
# def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
#     collate_fn = get_collate_fn(pad_index)
#     data_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         collate_fn=collate_fn,
#         shuffle=shuffle,
#     )
#     return data_loader
# # Dataloader

# batch_size = 512

# train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
# valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
# test_data_loader = get_data_loader(test_data, batch_size, pad_index)





# 2. 모델 구축 및 훈련
# **2-2. Attention**
# Attention Encoder

class AttentionEncoder(nn.Module):
    def __init__(
        self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )

        # outputs = [src length, batch size, encoder hidden dim * 2]
        # hidden = [batch size, decoder hidden dim]

        return outputs, hidden
class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim
        )
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_length = encoder_outputs.shape[0]
        # repeat decoder hidden state src_length times
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v_fc(energy).squeeze(2)

        return torch.softmax(attention, dim=1)
# Attention Decoder

class AttentionDecoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        encoder_hidden_dim,
        decoder_hidden_dim,
        dropout,
        attention,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU((encoder_hidden_dim * 2) + embedding_dim, decoder_hidden_dim)
        self.fc_out = nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim + embedding_dim, output_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # this also means that output == hidden
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]
        return prediction, hidden.squeeze(0), a.squeeze(1)
### Seq2seq + Attention 모델 class
# Seq2seq + Attention 연결한 모델
class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio):
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = src.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_length):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
        return outputs
# Seq2seq + Attention 모델 구축

input_dim = len(ko_vocab)
output_dim = len(en_vocab)
encoder_embedding_dim = 128
decoder_embedding_dim = 128
encoder_hidden_dim = 128
decoder_hidden_dim = 128
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention = Attention(encoder_hidden_dim, decoder_hidden_dim)

encoder = AttentionEncoder(
    input_dim,
    encoder_embedding_dim,
    encoder_hidden_dim,
    decoder_hidden_dim,
    encoder_dropout,
)

decoder = AttentionDecoder(
    output_dim,
    decoder_embedding_dim,
    encoder_hidden_dim,
    decoder_hidden_dim,
    decoder_dropout,
    attention,
)

model = AttentionSeq2Seq(encoder, decoder, device).to(device)
# 모델 가중치 초기화

def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)
# 훈련되는 parameter 수 출력(모든 parameter가 훈련되지 않을 수 있음)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")
### Seq2seq + Attention 훈련

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(data_loader)):
        src = batch["ko_ids"].to(device)
        trg = batch["en_ids"].to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)
def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            src = batch["ko_ids"].to(device)
            trg = batch["en_ids"].to(device)

            output = model(src, trg, 0)  # turn off teacher forcing
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)
n_epochs = 1
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")
for epoch in tqdm(range(n_epochs)):
    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "model_attention.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

model.load_state_dict(torch.load("model_attention.pt"))

test_loss = evaluate_fn(model, test_data_loader, criterion, device)

print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")




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

#     src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device).transpose(0,1)
    
#     # src_mask = model.make_src_mask(src_tensor)
    
#     with torch.no_grad():
#         # enc_src = model.encoder(src_tensor, src_mask)
#         encoder_outputs, hidden = model.encoder(src_tensor)

#     # trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
#     trg_indexes = en_vocab.lookup_indices([sos_token])

#     for i in range(max_len):

#         trg_tensor = torch.LongTensor(trg_indexes).to(device)
        
#         # trg_mask = model.make_trg_mask(trg_tensor)
#         with torch.no_grad():
#             # output, attention, = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
#             output, attention, _ = model.decoder(trg_tensor,hidden, encoder_outputs, )
        
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


# sen_list = [
# '모든 액체 , 젤 , 에어로졸 등 은 1 커트 짜리 여닫이 투명 봉지 하나 에 넣 어야 하 ㅂ니다 .',
# '미안 하 지만 , 뒷쪽 아이 들 의 떠들 는 소리 가 커 어서 , 광화문 으로 가 아고 싶 은데 표 를 바꾸 어 주 시 겠 어요 ?',
# '은행 이 너무 멀 어서 안 되 겠 네요 . 현찰 이 필요 하면 돈 을 훔치 시 어요',
# '아무래도 분실 하 ㄴ 것 같 으니 분실 신고서 를 작성 하 아야 하 겠 습니다 . 사무실 로 같이 가 시 ㄹ 까요 ?',
# '부산 에서 코로나 확진자 가 급증 하 아서 병상 이 부족하 아 지자  확진자 20명 을 대구 로 이송하 ㄴ다 .',
# '변기 가 막히 었 습니다 .',
# '그 바지 좀 보이 어 주 시 ㅂ시오 . 이거 얼마 에 사 ㄹ 수 있 는 것 이 ㅂ니까 ?',
# '비 가 오 아서 백화점 으로 가지 말 고 두타 로 가 았 으면 좋 겠 습니다 .',
# '속 이 안 좋 을 때 는 죽 이나 미음 으로 아침 을 대신 하 ㅂ니다',
# '문 대통령 은 집단 이익 에서 벗어 나 아 라고 말 하 었 다 .',
# '이것 좀 먹어 보 ㄹ 몇 일 간 의 시간 을 주 시 어요 .',
# '이날 개미군단 은 외인 의 물량 을 모두 받 아 내 었 다 .',
# '통합 우승 의 목표 를 달성하 ㄴ NC 다이노스 나성범 이 메이저리그 진출 이라는 또 다른 꿈 을 향하 어 나아가 ㄴ다 .',
# '이번 구조 조정 이 제품 을 효과 적 으로 개발 하 고 판매 하 기 위하 ㄴ 회사 의 능력 강화 조처 이 ㅁ 을 이해 하 아 주 시 리라 생각 하 ㅂ니다 .',
# '요즘 이 프로그램 녹화 하 며 많은 걸 느끼 ㄴ다 ']

