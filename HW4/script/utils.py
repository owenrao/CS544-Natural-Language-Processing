import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from collections import Counter

def create_dic(corpus, thres=5):
  counter = Counter([word for sent in corpus for word in sent])
  if not thres:
    result = {item[0]:id+1 for id,item in enumerate(counter.most_common())}
  else:
    result = {item[0]:id+1 for id,item in enumerate(counter.most_common()) if item[1]>thres}
  result['<pad>'] = 0
  return result

def collate_fn_padd(batch):
  data,label,lengths = zip(*batch)
  padded_data = pad_sequence(data, batch_first=True)
  padded_label = pad_sequence(label, batch_first=True)
  #print(padded_data.shape, padded_label.shape)
  return padded_data, padded_label, torch.tensor(lengths)

def read_data(dir):
  file_dir = dir
  sentence_list = []
  tag_list = []
  with open(file_dir) as file:
    sentence = []
    tags = []
    for line in file.readlines():
      line = line.strip().split()
      if len(line) == 0:
        sentence_list.append(sentence)
        tag_list.append(tags)
        sentence = []
        tags = []
        continue
      if str(dir).endswith("test"):
        tag = ''
      else:
        tag = line[-1]
      index, word = line[:2]
      sentence.append(word)
      tags.append(tag)
  return sentence_list, tag_list

def gen_emb(dir):
  print("Generating embedings..")
  glove_vocab = {} # word2id in glove 
  glove_emb = [] # array of embeddings, with index i corresponding to the word x glove_vocab[x] = i
  count = 4 # Give space for prior embeddings for pad, unks
  with open(dir, encoding='utf-8') as file:
      for line in file.readlines():
          line = line.strip().split()
          word, value = line[0], np.array(line[1:]+[0]).astype(float) # add another dimension for capitalization
          glove_vocab[word] = count
          glove_emb.append(value)
          count += 1
          # Add capitalized word
          cap_word = word.capitalize()
          if cap_word != word and cap_word not in glove_vocab.keys():
              glove_vocab[cap_word] = count
              cap_value = value.copy()
              cap_value[-1] = 1
              glove_emb.append(cap_value)
              count += 1
          # Add upper word
          upper_word = word.upper()
          if upper_word != word and upper_word not in glove_vocab.keys():
              glove_vocab[upper_word] = count
              upper_value = value.copy()
              upper_value[-1] = 2
              glove_emb.append(upper_value)
              count += 1
  glove_emb = np.array(glove_emb)
  assert len(glove_vocab)==len(glove_emb)
  # Add pad and unk emb
  pad_emb = np.zeros((1, glove_emb.shape[1])) # zero for padding
  unk_emb = np.mean(glove_emb, axis=0) # mean for unknown
  unk_emb[-1] = 0
  unk_cap_emb = unk_emb.copy()
  unk_cap_emb[-1] = 1
  unk_up_emb = unk_emb.copy()
  unk_up_emb[-1] = 2
  glove_vocab["< pad >"] = 0
  glove_vocab["< unk >"] = 1
  glove_vocab["< cap_unk >"] = 2
  glove_vocab["< up_unk >"] = 3
  glove_emb = np.vstack((pad_emb,unk_emb, unk_cap_emb,unk_up_emb,glove_emb))
  return glove_emb,glove_vocab

def gen_pred_list(model,loader,id2tag,device):
  pred_list = []
  model.eval()
  with torch.no_grad():
    predictions = []
    for data,labels,lengths in loader:
      data = data.to(device)
      labels = labels.to(device)
      pred = model(data,lengths)
      pred_list += [[id2tag[i] for i in p[:len]] for p,len in zip(pred.argmax(-1).tolist(),lengths.tolist())]
  return pred_list

def output_result(sentence_list, tag_list, predction_list, output_dir, test=False):
  print(f"Writing result to {output_dir}..")
  with open(output_dir,"w") as file:
    if test:
      for sentence,preds in zip(sentence_list, predction_list):
        index = 1
        for w,p in zip(sentence,preds):
          file.write(f"{index} {w} {p}\n")
          index += 1
    else:
      for sentence,tags,preds in zip(sentence_list, tag_list, predction_list):
        index = 1
        for w,t,p in zip(sentence,tags,preds):
          file.write(f"{index} {w} {t} {p}\n")
          index += 1
    print("Complete")
    