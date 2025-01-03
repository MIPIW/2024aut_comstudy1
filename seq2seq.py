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



## seq2seq

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
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


# train seq2seq

input_dim = len(ko_vocab)
output_dim = len(en_vocab)
# encoder_embedding_dim = 256
# decoder_embedding_dim = 256
# hidden_dim = 512
encoder_embedding_dim = 128
decoder_embedding_dim = 128
hidden_dim = 128
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


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
        src = batch["ko_ids"].to(device)
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
            src = batch["ko_ids"].to(device)
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
        torch.save(model.state_dict(), "tut1-model.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")


# evaluate seq2seq

model.load_state_dict(torch.load("tut1-model.pt"))

test_loss = evaluate_fn(model, test_data_loader, criterion, device)

print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

# def translate_sentence(
#     sentence,
#     model,
#     en_nlp,
#     ko_nlp,
#     en_vocab,
#     ko_vocab,
#     lower,
#     sos_token,
#     eos_token,
#     device,
#     max_output_length=25,
# ):
#     model.eval()
#     with torch.no_grad():
#         if isinstance(sentence, str):
#             tokens = [token.text for token in ko_nlp.tokenizer(sentence)]
#         else:
#             tokens = [token for token in sentence]
#         if lower:
#             tokens = [token.lower() for token in tokens]
#         tokens = [sos_token] + tokens + [eos_token]
#         ids = ko_vocab.lookup_indices(tokens)
#         tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
#         hidden, cell = model.encoder(tensor)
#         inputs = en_vocab.lookup_indices([sos_token])
#         for _ in range(max_output_length):
#             inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
#             output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
#             predicted_token = output.argmax(-1).item()
#             inputs.append(predicted_token)
#             if predicted_token == en_vocab[eos_token]:
#                 break
#         tokens = en_vocab.lookup_tokens(inputs)
#     return tokens

# sentence = test_data[0]["ko"]
# expected_translation = test_data[0]["en"]


# translation = translate_sentence(
#     sentence,
#     model,
#     en_nlp,
#     ko_nlp,
#     en_vocab,
#     ko_vocab,
#     lower,
#     sos_token,
#     eos_token,
#     device,
# )

# print(translation)
# sentence = "알파고 그거 망치로 치면 한 방이다."

# translation = translate_sentence(
#     sentence,
#     model,
#     en_nlp,
#     ko_nlp,
#     en_vocab,
#     ko_vocab,
#     lower,
#     sos_token,
#     eos_token,
#     device,
# )
# print(translation)


# translations = [
#     translate_sentence(
#         example["ko"],
#         model,
#         en_nlp,
#         ko_nlp,
#         en_vocab,
#         ko_vocab,
#         lower,
#         sos_token,
#         eos_token,
#         device,
#     )
#     for example in tqdm.tqdm(test_data)
# ]
# bleu = evaluate.load("bleu")
# predictions = [" ".join(translation[1:-1]) for translation in translations]
# references = [[example["en"]] for example in test_data]


# def get_tokenizer_fn(nlp, lower):
#     def tokenizer_fn(s):
#         tokens = [token.text for token in nlp.tokenizer(s)]
#         if lower:
#             tokens = [token.lower() for token in tokens]
#         return tokens

#     return tokenizer_fn
# tokenizer_fn = get_tokenizer_fn(en_nlp, lower)
# tokenizer_fn(predictions[0]), tokenizer_fn(references[0][0])
# results = bleu.compute(
#     predictions=predictions, references=references, tokenizer=tokenizer_fn
# )

# print(results)