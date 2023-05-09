import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
import numpy as np

class NER(Dataset):
    
    def __init__(self, sentence_list, tag_list, word2id, tag2id, transform=None):
        self.sentence_list = sentence_list
        self.tag_list = tag_list
        self.transform = transform
        self.word2id = word2id
        self.tag2id = tag2id
        #self.total_tag_count = len(tag2id)
        self.lengths = [len(sent) for sent in self.sentence_list]
        self.X = [torch.tensor([self.mapword2id(word) for word in sentence]) for sentence in self.sentence_list]
        self.y = [torch.tensor([self.tag2id.get(tag,0) for tag in tags]) for tags in self.tag_list]

    def mapword2id(self,word):
        try:
            return self.word2id[word]
        except KeyError:
            if word.isupper():
                return self.word2id['< up_unk >']
            elif word[0].isupper():
                return self.word2id['< cap_unk >']
            else:
                return self.word2id['< unk >']
        
    def __len__(self):
        return len(self.sentence_list)
    
    def __getitem__(self, index):
        emb = self.X[index] #.astype('float32').reshape(())
        unpad_len = self.lengths[index]
        label = self.y[index] #F.one_hot(label, num_classes=self.tag_len) #.float()
        
        if self.transform is not None:
            emb = self.transform(emb)
            
        return emb, label, unpad_len

class GLoVe_NER(Dataset):
    
    def __init__(self, sentence_list, tag_list, word2id, tag2id, transform=None):
      self.sentence_list = sentence_list
      self.tag_list = tag_list
      self.transform = transform
      self.word2id = word2id
      self.tag2id = tag2id
      #self.total_tag_count = len(tag2id)
      self.lengths = [len(sent) for sent in self.sentence_list]
      self.X = [torch.tensor([self.mapword2id(word) for word in sentence]) for sentence in self.sentence_list]
      self.y = [torch.tensor([self.tag2id.get(tag,0) for tag in tags]) for tags in self.tag_list]

    def mapword2id(self,word):
      try:
        return self.word2id[word]
      except KeyError:
        if word.isupper():
          return self.word2id['< up_unk >']
        elif word[0].isupper():
          return self.word2id['< cap_unk >']
        else:
          return self.word2id['< unk >']
    
    def __len__(self):
      return len(self.sentence_list)
    
    def __getitem__(self, index):
      emb = self.X[index] #.astype('float32').reshape(())
      unpad_len = self.lengths[index]
      label = self.y[index] #F.one_hot(label, num_classes=self.tag_len) #.float()
      
      if self.transform is not None:
          emb = self.transform(emb)
          
      return emb, label, unpad_len

class BiLSTM(nn.Module):
    def __init__(self, device, vocab_size, hidden_size, output_size, emb_size = 100, n_layers=1, dropout=0.33, emb=None):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb_layer = nn.Embedding(vocab_size, emb_size, padding_idx=0, device=device)
        self.emb_layer.weight.data.uniform_(-np.sqrt(1/emb_size), np.sqrt(1/emb_size))
        self.encoder_layer = nn.LSTM(emb_size, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True, device=device)
        self.linear_layer1 = nn.Linear(hidden_size*2, 128, device=device)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer2 = nn.Linear(128, output_size, device=device)
        self.elu = nn.ELU()

    def forward(self, x, x_lengths):
        x = self.emb_layer(x)
        #print(x.shape)
        x = pack_padded_sequence(x, lengths=x_lengths, batch_first=True, enforce_sorted=False)
        x,_ = self.encoder_layer(x)
        #print(x.data.shape)
        x,_ = pad_packed_sequence(x, batch_first=True)
        #print(x.shape)
        x = self.dropout(x)
        x = self.linear_layer1(x)
        x = self.dropout(x)
        x = self.elu(x)
        x = self.linear_layer2(x)
        x = self.dropout(x)
        return x

class BiLSTM_GLoVe(nn.Module):
    def __init__(self, device, emb, vocab_size, hidden_size, output_size, emb_size = 100, n_layers=1, dropout=0.33):
        super(BiLSTM_GLoVe, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.emb_layer = nn.Embedding(vocab_size, emb_size, padding_idx=0, device=device)
        self.emb_layer.load_state_dict({'weight': torch.from_numpy(emb).float()})
        self.emb_layer.weight.requires_grad = False
        self.encoder_layer = nn.LSTM(emb_size, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True, device=device)
        self.linear_layer1 = nn.Linear(hidden_size*2, 128, device=device)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer2 = nn.Linear(128, output_size, device=device)
        self.elu = nn.ELU()

    def forward(self, x, x_lengths):
        '''
        idx_sort = torch.argsort(-x_lengths)
        idx_unsort = torch.argsort(idx_sort)
        x = x[idx_sort]
        x_lengths = x_lengths[idx_sort]
        '''
        x = self.emb_layer(x)
        #print(x.shape)
        x = pack_padded_sequence(x, lengths=x_lengths, batch_first=True, enforce_sorted=False)
        x,_ = self.encoder_layer(x)
        #print(x.data.shape)
        x,_ = pad_packed_sequence(x, batch_first=True)
        #print(x.shape)
        x = self.dropout(x)
        x = self.linear_layer1(x)
        x = self.dropout(x)
        x = self.elu(x)
        x = self.linear_layer2(x)
        x = self.dropout(x)
        return x



