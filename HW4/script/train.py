import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from model import *
from utils import *

def eval(model, loader, criterion, device):
  test_loss = 0.0
  model.eval()
  for data, label, lengths in loader:
    data = data.to(device)
    label = label.to(device)
    pred = model(data, lengths)
    #label = F.one_hot(label, num_classes=10).float()
    loss = criterion(torch.permute(pred,(0,2,1)), label)
    test_loss += loss.item()*data.size(0)
    
  test_loss = test_loss/len(loader.dataset)
  return test_loss

def train(model, train_loader, val_loader, epoch_num, device, lr=0.5, step_size=5, gamma=1, momentum=0.8):
  criterion = nn.CrossEntropyLoss()
  weight_decay = 1e-4
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
  #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=False)
  best_val_loss = np.inf
  for epoch in range(epoch_num):
    train_loss = 0.0
    model.train()
    for data, label, lengths in train_loader:
      data = data.to(device)
      label = label.to(device)
      #label = F.one_hot(label, num_classes=10).float()
      optimizer.zero_grad()
      pred = model(data, lengths)
      #print(pred.shape,label.shape)
      loss = criterion(torch.permute(pred,(0,2,1)), label)
      loss.backward()

      #indices = torch.LongTensor([-1])
      #model.emb_layer.weight.grad[indices] = 0
      
      optimizer.step()
      train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    val_loss = eval(model, val_loader, criterion, device)
    if epoch_num>5 and (epoch+1)%5 == 0:
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss}")
        print(f"[Epoch {epoch+1}] Val Loss: {val_loss}")
    if val_loss<best_val_loss:
      best_val_loss = val_loss
      torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, 'best_model.pth')
    scheduler.step()
  #print(best_val_loss)
  checkpoint = torch.load('best_model.pth')
  model.load_state_dict(checkpoint['model_state_dict'])
  return model

if __name__ == "__main__":
    import sys
    import os
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_DIR = sys.argv[1]
    MODEL = sys.argv[2]
    MODEL_FOLDER_PATH = os.path.join(os.path.dirname(CURRENT_PATH),"models")
    try:
        EMB_DIR = sys.argv[3]
    except:
        EMB_DIR = None

    assert os.path.exists(os.path.join(DATA_DIR,"train"))
    assert os.path.exists(os.path.join(DATA_DIR,"dev"))

    # Data 
    train_sentences, train_tags = read_data(os.path.join(DATA_DIR,"train"))
    dev_sentences, dev_tags = read_data(os.path.join(DATA_DIR,"dev"))

    tag2id = create_dic(train_tags, thres=None)
    id2tag = {v:k for k,v in tag2id.items()}
    tag_count = len(tag2id)

    if MODEL == "blstm1":
        # build vocab
        word2id = create_dic(train_sentences, thres=2)
        word2id['< unk >'] = len(word2id)
        word2id['< up_unk >'] = len(word2id)
        word2id['< cap_unk >'] = len(word2id)
        id2word = {v:k for k,v in word2id.items()}
        vocab_size = len(word2id)
        # build dataloader
        batch_size = 32
        train_ds = NER(train_sentences, train_tags, word2id, tag2id)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,\
            shuffle=True, collate_fn=collate_fn_padd)
        val_ds = NER(dev_sentences, dev_tags, word2id, tag2id)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,\
            shuffle=False, collate_fn=collate_fn_padd)
        # import model
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        model = BiLSTM(device, vocab_size, 256, tag_count).to(device)
        # Train
        print("Training BLSTM1 model..")
        model = train(model, train_loader, val_loader, 50, device, lr=0.5, step_size=5, gamma=0.8)
        print("Training complete")
        # save model
        if not os.path.exists(MODEL_FOLDER_PATH):
            os.mkdir("models")
        torch.save(model.state_dict(), os.path.join(MODEL_FOLDER_PATH,"blstm1.pt"))
        print(f"Model saved at {MODEL_FOLDER_PATH}")
        print("If want to predict using trained model, plz use predict.py file.")

    elif MODEL == "blstm2":
        assert EMB_DIR
        # Gen Emb matrix
        glove_emb,glove_vocab = gen_emb(EMB_DIR)
        id2word = {v:k for k,v in glove_vocab.items()}
        # build dataloader
        batch_size = 32
        train_ds = GLoVe_NER(train_sentences, train_tags, glove_vocab, tag2id)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,\
            shuffle=True, collate_fn=collate_fn_padd)
        val_ds = GLoVe_NER(dev_sentences, dev_tags, glove_vocab, tag2id)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,\
            shuffle=False, collate_fn=collate_fn_padd)
        # import model
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        model = BiLSTM_GLoVe(device, glove_emb, glove_emb.shape[0], 256, tag_count, emb_size=glove_emb.shape[1]).to(device)
        # Train
        print("Training BLSTM2 model..")
        model = train(model, train_loader, val_loader, 50, device, lr=0.6, step_size=8, gamma=0.8)
        print("Training complete")
        # save mode
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(model.state_dict(), os.path.join(MODEL_FOLDER_PATH,"blstm2.pt"))
        print(f"Model saved at {MODEL_FOLDER_PATH}")
        print("If want to predict using trained model, plz use predict.py file.")
    else:
        print("Wrong input model. Please choose from 'blstm1' and 'blstm2' as input.")