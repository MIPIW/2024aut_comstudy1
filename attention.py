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
import os 
os.environ['CUDA_LAUNCH_BLOCKING']=1

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
    
max_length = 32
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

batch_size = 32
train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################################

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# import spacy
# import datasets
# import torchtext
# import tqdm
# import evaluate
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# seed = 1234

# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# dataset = datasets.load_dataset("bentrevett/multi30k")

# train_data, valid_data, test_data = (
#     dataset["train"],
#     dataset["validation"],
#     dataset["test"],
# )
# en_nlp = spacy.load("en_core_web_sm")
# de_nlp = spacy.load("de_core_news_sm")
# def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
#     en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
#     de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])][:max_length]
#     if lower:
#         en_tokens = [token.lower() for token in en_tokens]
#         de_tokens = [token.lower() for token in de_tokens]
#     en_tokens = [sos_token] + en_tokens + [eos_token]
#     de_tokens = [sos_token] + de_tokens + [eos_token]
#     return {"en_tokens": en_tokens, "de_tokens": de_tokens}
# max_length = 1_000
# lower = True
# sos_token = "<sos>"
# eos_token = "<eos>"

# fn_kwargs = {
#     "en_nlp": en_nlp,
#     "de_nlp": de_nlp,
#     "max_length": max_length,
#     "lower": lower,
#     "sos_token": sos_token,
#     "eos_token": eos_token,
# }

# train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
# valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
# test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)
# min_freq = 2
# unk_token = "<unk>"
# pad_token = "<pad>"

# special_tokens = [
#     unk_token,
#     pad_token,
#     sos_token,
#     eos_token,
# ]

# en_vocab = torchtext.vocab.build_vocab_from_iterator(
#     train_data["en_tokens"],
#     min_freq=min_freq,
#     specials=special_tokens,
# )

# de_vocab = torchtext.vocab.build_vocab_from_iterator(
#     train_data["de_tokens"],
#     min_freq=min_freq,
#     specials=special_tokens,
# )
# assert en_vocab[unk_token] == de_vocab[unk_token]
# assert en_vocab[pad_token] == de_vocab[pad_token]

# unk_index = en_vocab[unk_token]
# pad_index = en_vocab[pad_token]
# en_vocab.set_default_index(unk_index)
# de_vocab.set_default_index(unk_index)
# def numericalize_example(example, en_vocab, de_vocab):
#     en_ids = en_vocab.lookup_indices(example["en_tokens"])
#     de_ids = de_vocab.lookup_indices(example["de_tokens"])
#     return {"en_ids": en_ids, "de_ids": de_ids}
# fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

# train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
# valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
# test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)
# data_type = "torch"
# format_columns = ["en_ids", "de_ids"]

# train_data = train_data.with_format(
#     type=data_type, columns=format_columns, output_all_columns=True
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
# def get_collate_fn(pad_index):
#     def collate_fn(batch):
#         batch_en_ids = [example["en_ids"] for example in batch]
#         batch_de_ids = [example["de_ids"] for example in batch]
#         batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
#         batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
#         batch = {
#             "en_ids": batch_en_ids,
#             "de_ids": batch_de_ids,
#         }
#         return batch

#     return collate_fn
# def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
#     collate_fn = get_collate_fn(pad_index)
#     data_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         collate_fn=collate_fn,
#         shuffle=shuffle,
#     )
#     return data_loader
# batch_size = 128

# train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
# valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
# test_data_loader = get_data_loader(test_data, batch_size, pad_index)







class Encoder(nn.Module):
    def __init__(
        self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, hidden = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer
        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
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
        # hidden = [batch size, decoder hidden dim]
        # encoder_outputs = [src length, batch size, encoder hidden dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_length = encoder_outputs.shape[0]
        # repeat decoder hidden state src_length times
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src length, decoder hidden dim]
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src length, decoder hidden dim]
        attention = self.v_fc(energy).squeeze(2)
        # attention = [batch size, src length]
        return torch.softmax(attention, dim=1)
class Decoder(nn.Module):
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
        # input = [batch size]
        # hidden = [batch size, decoder hidden dim]
        # encoder_outputs = [src length, batch size, encoder hidden dim * 2]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src length]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src length]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, encoder hidden dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, encoder hidden dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, (encoder hidden dim * 2) + embedding dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [seq length, batch size, decoder hid dim * n directions]
        # hidden = [n layers * n directions, batch size, decoder hid dim]
        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, decoder hidden dim]
        # hidden = [1, batch size, decoder hidden dim]
        # this also means that output == hidden
        assert (output == hidden).all()
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch size, output dim]
        return prediction, hidden.squeeze(0), a.squeeze(1)
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
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
        # outputs = [src length, batch size, encoder hidden dim * 2]
        # hidden = [batch size, decoder hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, decoder hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs
# input_dim = len(de_vocab)
input_dim = len(ko_vocab)
output_dim = len(en_vocab)
encoder_embedding_dim = 4
decoder_embedding_dim = 4
encoder_hidden_dim = 4
decoder_hidden_dim = 4
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention = Attention(encoder_hidden_dim, decoder_hidden_dim)

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    encoder_hidden_dim,
    decoder_hidden_dim,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    encoder_hidden_dim,
    decoder_hidden_dim,
    decoder_dropout,
    attention,
)

model = Seq2Seq(encoder, decoder, device).to(device)
def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["de_ids"].to(device)
        trg = batch["en_ids"].to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
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
        for i, batch in enumerate(data_loader):
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

n_epochs = 1
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

for epoch in tqdm.tqdm(range(n_epochs)):
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
        torch.save(model.state_dict(), "tut3-model.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")
model.load_state_dict(torch.load("tut3-model.pt"))

test_loss = evaluate_fn(model, test_data_loader, criterion, device)

print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")
def translate_sentence(
    sentence,
    model,
    en_nlp,
    de_nlp,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            de_tokens = [token.text for token in de_nlp.tokenizer(sentence)]
        else:
            de_tokens = [token for token in sentence]
        if lower:
            de_tokens = [token.lower() for token in de_tokens]
        de_tokens = [sos_token] + de_tokens + [eos_token]
        ids = de_vocab.lookup_indices(de_tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        encoder_outputs, hidden = model.encoder(tensor)
        inputs = en_vocab.lookup_indices([sos_token])
        attentions = torch.zeros(max_output_length, 1, len(ids))
        for i in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, attention = model.decoder(
                inputs_tensor, hidden, encoder_outputs
            )
            attentions[i] = attention
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
        en_tokens = en_vocab.lookup_tokens(inputs)
    return en_tokens, de_tokens, attentions[: len(en_tokens) - 1]


sentence = test_data[0]["de"]
expected_translation = test_data[0]["en"]

sentence, expected_translation
translation, sentence_tokens, attention = translate_sentence(
    sentence,
    model,
    en_nlp,
    ko_nlp,
    en_vocab,
    ko_vocab,
    lower,
    sos_token,
    eos_token,
    device,
)
translation, sentence_tokens, attention = translate_sentence(
    sentence,
    model,
    en_nlp,
    ko_nlp,
    en_vocab,
    ko_vocab,
    lower,
    sos_token,
    eos_token,
    device,
)
translation
translations = [
    translate_sentence(
        example["de"],
        model,
        en_nlp,
        ko_nlp,
        en_vocab,
        ko_vocab,
        lower,
        sos_token,
        eos_token,
        device,
    )[0]
    for example in tqdm.tqdm(test_data)
]
bleu = evaluate.load("bleu")
predictions = [" ".join(translation[1:-1]) for translation in translations]

references = [[example["en"]] for example in test_data]
def get_tokenizer_fn(nlp, lower):
    def tokenizer_fn(s):
        tokens = [token.text for token in nlp.tokenizer(s)]
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens

    return tokenizer_fn
tokenizer_fn = get_tokenizer_fn(en_nlp, lower)
results = bleu.compute(
    predictions=predictions, references=references, tokenizer=tokenizer_fn
)
print(results)