import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from model import *
from utils import *

if __name__ == "__main__":
    import os
    import sys
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    DATA_DIR = sys.argv[1]
    MODEL = sys.argv[2]
    MODEL_DIR = sys.argv[3]
    OUTPUT_DIR = os.path.join(os.path.dirname(CURRENT_PATH),"result")
    try:
        EMB_DIR = sys.argv[4]
    except:
        EMB_DIR = None

    assert os.path.exists(os.path.join(DATA_DIR,"test"))
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    # Data 
    train_sentences, train_tags = read_data(os.path.join(DATA_DIR,"train"))
    dev_sentences, dev_tags = read_data(os.path.join(DATA_DIR,"dev"))
    test_sentences, _ = read_data(os.path.join(DATA_DIR,"test"))

    tag2id = create_dic(train_tags, thres=None)
    id2tag = {v:k for k,v in tag2id.items()}
    tag_count = len(tag2id)

    if MODEL == "blstm1":
        word2id = create_dic(train_sentences, thres=2)
        word2id['< unk >'] = len(word2id)
        word2id['< up_unk >'] = len(word2id)
        word2id['< cap_unk >'] = len(word2id)
        id2word = {v:k for k,v in word2id.items()}
        vocab_size = len(word2id)
        # build dataloader
        batch_size = 64
        val_ds = NER(dev_sentences, dev_tags, word2id, tag2id)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,\
            shuffle=False, collate_fn=collate_fn_padd)
        test_ds = NER(test_sentences, [[0 for word in sent] for sent in test_sentences], word2id, tag2id)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,\
            shuffle=False, collate_fn=collate_fn_padd)
        # import model
        model = BiLSTM(device, vocab_size, 256, tag_count).to(device)
        model.load_state_dict(torch.load(MODEL_DIR))
        # gen predictions
        dev_pred_list = gen_pred_list(model,val_loader,id2tag,device)
        test_pred_list = gen_pred_list(model,test_loader,id2tag,device)
        # save output
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        output_result(test_sentences, None, test_pred_list, os.path.join(OUTPUT_DIR,"test1.out"), test=True)
        output_result(test_sentences, dev_tags, dev_pred_list, os.path.join(OUTPUT_DIR,"dev1.out"), test=False)
    elif MODEL == "blstm2":
        assert EMB_DIR
        glove_emb,glove_vocab = gen_emb(EMB_DIR)
        id2word = {v:k for k,v in glove_vocab.items()}
        batch_size = 64
        val_ds = GLoVe_NER(dev_sentences, dev_tags, glove_vocab, tag2id)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,\
            shuffle=False, collate_fn=collate_fn_padd)
        test_ds = GLoVe_NER(test_sentences, [[0 for word in sent] for sent in test_sentences], glove_vocab, tag2id)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
            shuffle=False, collate_fn=collate_fn_padd)
        # import model
        model = BiLSTM_GLoVe(device, glove_emb, glove_emb.shape[0], 256, tag_count, emb_size=glove_emb.shape[1]).to(device)
        model.load_state_dict(torch.load(MODEL_DIR))
        # gen predictions
        dev_pred_list = gen_pred_list(model,val_loader,id2tag,device)
        test_pred_list = gen_pred_list(model,test_loader,id2tag,device)
        # save output
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        output_result(test_sentences, None, test_pred_list, os.path.join(OUTPUT_DIR,"test2.out"), test=True)
        output_result(test_sentences, dev_tags, dev_pred_list, os.path.join(OUTPUT_DIR,"dev2.out"), test=False)