from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pickle
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
import math
from torchtext.data.metrics import bleu_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from evaluate import load
import pickle
from nltk.translate.meteor_score import single_meteor_score 


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

def yield_tokens_en(file_path, ln):
    file = open(file_path, 'r')

    for line in file:
        yield token_transform[ln](eval(line)[ln])

vocab_transform['de'] = build_vocab_from_iterator(yield_tokens_en("train_small.txt", 'de'), min_freq=1, specials=special_symbols, special_first=True)
vocab_transform['en'] = build_vocab_from_iterator(yield_tokens_en("train_small.txt", 'en'), min_freq=1, specials=special_symbols, special_first=True)

vocab_transform['de'].set_default_index(UNK_IDX)
vocab_transform['en'].set_default_index(UNK_IDX)
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):

        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size, dim_feedforward = 512, dropout = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
saved_model_path = 'MT_de_en_20.pth'
checkpoint = torch.load(saved_model_path, map_location=torch.device('cpu'))
transformer.load_state_dict(checkpoint['model_state_dict'])
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),torch.tensor(token_ids), torch.tensor([EOS_IDX])))

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], vocab_transform[ln], tensor_transform) 

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src
    src_mask = src_mask
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long)

    for i in range(max_len-1):
        memory = memory
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool))
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

print(translate(transformer, "Wie Sie sicher aus der Presse und dem Fernsehen wissen, gab es in Sri Lanka mehrere Bombenexplosionen mit zahlreichen Toten."))

val_list_de = []
val_list_en = []
val_file = open('val.txt','r')

for line in val_file:
    temp_dict = eval(line)
    val_list_de.append(temp_dict['de'])
    val_list_en.append(temp_dict['en'])

translated_val_en = []
for i in val_list_de:
    translated_val_en.append(translate(transformer, i))

test_list_de = []
test_list_en = []

test_file = open('test.txt','r')
for line in test_file:
    temp_dict = eval(line)
    test_list_de.append(temp_dict['de'])
    test_list_en.append(temp_dict['en'])

translated_test_en = []
for i in test_list_de:
    translated_test_en.append(translate(transformer, i))


predictions = translated_val_en
references = val_list_en

bleu = load("bleu")

bleu1_score = bleu.compute(predictions=predictions, references=references, max_order=1)
bleu2_score = bleu.compute(predictions=predictions, references=references, max_order=2)
bleu3_score = bleu.compute(predictions=predictions, references=references, max_order=3)
bleu4_score = bleu.compute(predictions=predictions, references=references, max_order=4)

print("BLEU-1 for Val Set:", bleu1_score)
print("BLEU-2 for Val Set:", bleu2_score)
print("BLEU-3 for Val Set:", bleu3_score)
print("BLEU-4 for Val Set:", bleu4_score)

meteor = []
for i in range(len(predictions)):
    temp = single_meteor_score(predictions[i].split(), references[i].split())
    meteor.append(temp)
print("AVG METEOR: ", sum(meteor)/len(meteor))
'''
bert = load("bertscore")
bert_score = bert.compute(predictions=predictions, references=references, lang='en')
print("BERT AVERAGE of precision:", sum(bert_score['precision'])/len(bert_score['precision']))
print("BERT AVERAGE of recall:", sum(bert_score['recall'])/len(bert_score['recall']))
print("BERT AVERAGE of f1:", sum(bert_score['f1'])/len(bert_score['f1']))
'''
predictions = translated_test_en
references = test_list_en
bleu = load("bleu")

bleu1_score = bleu.compute(predictions=predictions, references=references, max_order=1)
bleu2_score = bleu.compute(predictions=predictions, references=references, max_order=2)
bleu3_score = bleu.compute(predictions=predictions, references=references, max_order=3)
bleu4_score = bleu.compute(predictions=predictions, references=references, max_order=4)

print("BLEU-1 for Test Set: ", bleu1_score)
print("BLEU-2 for Test Set: ", bleu2_score)
print("BLEU-3 for Test Set: ", bleu3_score)
print("BLEU-4 for Test Set: ", bleu4_score)

meteor = []
for i in range(len(predictions)):
    temp = single_meteor_score(predictions[i].split(), references[i].split())
    meteor.append(temp)
print("AVG METEOR: ", sum(meteor)/len(meteor))

bert = load("bertscore")
bert_score = bert.compute(predictions=predictions, references=references, lang='en')
print("BERT AVERAGE of precision:", sum(bert_score['precision'])/len(bert_score['precision']))
print("BERT AVERAGE of recall:", sum(bert_score['recall'])/len(bert_score['recall']))
print("BERT AVERAGE of f1:", sum(bert_score['f1'])/len(bert_score['f1']))
