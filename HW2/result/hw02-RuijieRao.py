import pandas as pd
import numpy as np
import json
from string import punctuation
from collections import Counter 
from os import path

thres = 2

def pseudo_words_encoding(word, filtered_vocab_list):
    if word in filtered_vocab_list:
        return word
    elif "-" in word:
        return "< -unk >"
    elif word.isupper():
        return "< allcap-unk >"
    elif word[0].isupper():
        return "< Title-unk >"
    else: return "< unk >"

def accuracy(prediction, ground_truth):
    return sum(np.array(prediction) == np.array(ground_truth))/len(ground_truth)

def greedy_decoding(test_df, initial_array, transition_matrix, emission_matrix, filtered_vocab_list, tag_list):
    result = []
    for index, row in test_df.iterrows():
        if row.position_index == 1:
            prev_state_index = 0
            t_vec = initial_array
        else:
            t_vec = transition_matrix[prev_state_index,:]
        word_index = filtered_vocab_list.index(pseudo_words_encoding(row.word, filtered_vocab_list))
        e_vec = emission_matrix[:,word_index]
        argmax_index = np.argmax(t_vec*e_vec.T)
        prev_state_index = argmax_index
        argmax_tag = tag_list[argmax_index]
        result.append(argmax_tag)
    return result

def create_corpus(words):
    corpus = []
    line = []
    for word in words:
        line.append(word)
        if word==".":
            corpus.append(line)
            line = []
    return corpus

class Viterbi:
    def __init__(self, t_dict, e_dict, vocab, tags) -> None:
        self.t_dict= t_dict
        self.e_dict = e_dict
        self.vocab = vocab
        self.tags = tags # without < start >
        self.K = len(self.tags)

    def predict(self, corpus:list[str]):
        N = len(corpus)
        M = np.zeros((N, self.K)) # Num of words x Num of tags
        states_record = np.zeros((N-1, self.K)) # Records the best previous states except for the initial
        for i, word_ in enumerate(corpus):
            word = pseudo_words_encoding(word_, self.vocab)
            for j in range(self.K):
                tag = self.tags[j]
                if i == 0: # initial state
                    M[i,j] = self.t_dict.get(('< start >', tag),0) * self.e_dict.get((tag, word),0)
                else:
                    temp = [M[i-1,k]*self.t_dict.get((tag_,tag),0)*self.e_dict.get((tag, word),0) for k,tag_ in enumerate(self.tags)]
                    best_prev_state_i = np.argmax(temp)
                    states_record[i-1,j] = best_prev_state_i
                    M[i,j] = temp[best_prev_state_i]
        # Backward
        result = []
        best_state_i = np.argmax(M[-1,:])
        result.append(self.tags[best_state_i])
        for i in range(len(states_record)-1,-1,-1):
            best_state_i = int(states_record[i,best_state_i])
            result.append(self.tags[best_state_i])
        return result[::-1]

def main(DATA_DIR="", OUTPUT_DIR="", decoding="both"):

    # Data Import
    train_df = pd.read_csv(path.join(DATA_DIR,"train"), sep="\t", names=["position_index","word","tag"])
    dev_df = pd.read_csv(path.join(DATA_DIR,"dev"), sep="\t", names=["position_index","word","tag"])
    test_df = pd.read_csv(path.join(DATA_DIR,"test"), sep="\t", names=["position_index","word"])

    # Vocab
    vocab_df = train_df.groupby("word").count()[["tag"]].rename(columns={'tag':'occurences'})\
        .reset_index()
    filtered_vocab = vocab_df[vocab_df["occurences"]>=thres]
    filtered_vocab = filtered_vocab.append({'word':"< unk >",'occurences':vocab_df[vocab_df["occurences"]<thres]["occurences"]\
        .sum()}, ignore_index=True).sort_values(by="occurences", ascending=False).reset_index(drop=True)
    filtered_vocab_list = filtered_vocab["word"].to_list()
    filtered_vocab_list += ["< -unk >", "< allcap-unk >", "< Title-unk >"]

    # Transform the training data
    train_df_filtered = train_df.copy()
    train_df_filtered["word"] = train_df["word"].apply(lambda x: pseudo_words_encoding(x,filtered_vocab_list))
    # T and E and I
    ## Count of States
    count_states = Counter(train_df_filtered["tag"].values)
    tag_list = [t for t in count_states.keys() if t != "< start >"]
    count_states["< start >"] = train_df_filtered[train_df_filtered["position_index"]==1]["word"].count()
    ## Count of s -> s'
    transition_vocab = train_df_filtered.rename(columns={'tag':'state'}).copy()
    next_state = transition_vocab["state"].copy()
    transition_vocab["state'"] = next_state[1:].reset_index(drop=True)
    initial_selection = (transition_vocab["position_index"]==1)[1:].reset_index(drop=True)
    initial_selection.loc[len(initial_selection)] = True
    transition_vocab.loc[initial_selection, "state"] = "< start >"
    transition_vocab.loc[len(transition_vocab)-1, "state'"] = transition_vocab.loc[0, "state"]
    transition_count = transition_vocab.groupby(["state","state'"])[["word"]].count().reset_index()\
        .rename(columns={"word":"occurences"}).sort_values(by="occurences", ascending=False)
    ## Transition Prob
    def t_prob(x):
            return x.occurences/count_states[x.state]
    transition_prob = transition_count
    transition_prob["p"] = transition_prob.apply(t_prob, axis=1)
    #transition_prob["key"] = transition_prob.apply(lambda x: str((x.state,x["state'"])), axis=1)
    transition_prob["key"] = transition_prob.apply(lambda x: (x.state,x["state'"]), axis=1)
    transition_prob.set_index("key", inplace=True)
    transition_dict = transition_prob[["p"]].to_dict()["p"]
    ## Count of s -> x
    emission_vocab = train_df_filtered.rename(columns={'tag':'state'}).copy()
    emission_count = emission_vocab.groupby(["state","word"])[["position_index"]].count().reset_index()\
        .rename(columns={"position_index":"occurences"}).sort_values(by="occurences", ascending=False)
    ## Emission Prob
    emission_prob = emission_count
    emission_prob["key"] = emission_prob.apply(lambda x: (x.state,x.word), axis=1)
    emission_prob.set_index("key", inplace=True)
    emission_prob["p"] = emission_prob.apply(lambda x: x.occurences/count_states[x.state], axis=1)
    emission_dict = emission_prob[["p"]].to_dict()["p"]

    # Greedy
    if decoding == ("both" or "greedy"):
        K  = len(tag_list)
        V = len(filtered_vocab_list)
        transition_matrix = np.zeros((K,K))
        emission_matrix = np.zeros((K,V))
        initial_array = np.zeros((1,K))
        ## transition_matrix & initial_array
        for key, value in transition_dict.items():
            state, state_ = key
            j = tag_list.index(state_)
            if state == "< start >":
                initial_array[0,j] = value
                continue
            i = tag_list.index(state)
            transition_matrix[i,j] = value
        ## emission_matrix
        for key, value in emission_dict.items():
            state, word = key
            j = filtered_vocab_list.index(word)
            i = tag_list.index(state)
            emission_matrix[i,j] = value
        dev_pred = greedy_decoding(dev_df, initial_array, transition_matrix, emission_matrix, filtered_vocab_list, tag_list=tag_list)
        print("Accuracy with Greedy decoding for dev: ", accuracy(dev_pred, dev_df["tag"].values))
        ## Output
        test_df["greedy"] = greedy_decoding(test_df, initial_array, transition_matrix, emission_matrix, filtered_vocab_list, tag_list=tag_list)
        test_df[["position_index","word","greedy"]].to_csv(path.join(OUTPUT_DIR, "greedy.out"), header=False, index=False, sep="\t")

    # Viterbi
    if decoding == ("both" or "viterbi"):
        myViterbi = Viterbi(transition_dict, emission_dict, filtered_vocab_list, tag_list)
        dev_pred = []
        for sentence in create_corpus(dev_df["word"].values):
            dev_pred += myViterbi.predict(sentence)
        print("Accuracy with Viterbi decoding for dev: ", accuracy(dev_pred, dev_df["tag"].values))
        ## Output
        result = []
        for sentence in create_corpus(test_df["word"].values):
            result += myViterbi.predict(sentence)
        test_df["viterbi"] = result
        test_df[["position_index","word","viterbi"]].to_csv(path.join(OUTPUT_DIR, "viterbi.out"), header=False, index=False, sep="\t")
        
if __name__ == "__main__":
    import sys
    main(DATA_DIR=sys.argv[1], OUTPUT_DIR=sys.argv[2], decoding=sys.argv[3])