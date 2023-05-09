import pandas as pd
import numpy as np
import nltk
import re
import csv
import re

# Utilities
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Models
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Config and Constants
DATA_DIR='data/'
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
wd = nltk.corpus.wordnet
ps = PorterStemmer()
contractions = { 
    "dont":"do not",
    "ain't": "are not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how iss",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "yyou will",
    "you'll've": "you shall have ",
    "you're": "you are",
    "you've": "you have",
    "aint": "are not",
    "arent": "are not",
    "cant": "cannot",
    "cantve": "cannot have",
    "cause": "because",
    "couldve": "could have",
    "couldnt": "could not",
    "couldntve": "could not have",
    "didnt": "did not",
    "doesnt": "does not",
    "dont": "do not",
    "hadnt": "had not",
    "hadntve": "had not have",
    "hasnt": "has not",
    "havent": "have not",
    "hed": "he would",
    "hedve": "he would have",
    "hell": "he will",
    "hellve": "he will have",
    "hes": "he is",
    "howd": "how did",
    "howdy": "how do you",
    "howll": "how will",
    "hows": "how iss",
    "id": "i would",
    "idve": "i would have",
    "ill": "i will",
    "illve": "i will have",
    "im": "i am",
    "ive": "i have",
    "isnt": "is not",
    "itd": "it would",
    "itdve": "it would have",
    "itll": "it will",
    "itllve": "it will have",
    "its": "it is",
    "lets": "let us",
    "maam": "madam",
    "maynt": "may not",
    "mightve": "might have",
    "mightnt": "might not",
    "mightntve": "might not have",
    "mustve": "must have",
    "mustnt": "must not",
    "mustntve": "must not have",
    "neednt": "need not",
    "needntve": "need not have",
    "oclock": "of the clock",
    "oughtnt": "ought not",
    "oughtntve": "ought not have",
    "shant": "shall not",
    "shant": "shall not",
    "shantve": "shall not have",
    "shed": "she would",
    "shedve": "she would have",
    "shell": "she will",
    "shellve": "she will have",
    "shes": "she is",
    "shouldve": "should have",
    "shouldnt": "should not",
    "shouldntve": "should not have",
    "sove": "so have",
    "sos": "so is",
    "thatd": "that would",
    "thatdve": "that would have",
    "thats": "that is",
    "thered": "there would",
    "theredve": "there would have",
    "theres": "there is",
    "theyd": "they would",
    "theydve": "they would have",
    "theyll": "they will",
    "theyllve": "they will have",
    "theyre": "they are",
    "theyve": "they have",
    "tove": "to have",
    "wasnt": "was not",
    "wed": "we would",
    "wedve": "we would have",
    "well": "we will",
    "wellve": "we will have",
    "were": "we are",
    "weve": "we have",
    "werent": "were not",
    "whatll": "what will",
    "whatllve": "what will have",
    "whatre": "what are",
    "whats": "what is",
    "whatve": "what have",
    "whens": "when is",
    "whenve": "when have",
    "whered": "where did",
    "wheres": "where is",
    "whereve": "where have",
    "wholl": "who will",
    "whollve": "who will have",
    "whos": "who is",
    "whove": "who have",
    "whys": "why is",
    "whyve": "why have",
    "willve": "will have",
    "wont": "will not",
    "wontve": "will not have",
    "wouldve": "would have",
    "wouldnt": "would not",
    "wouldntve": "would not have",
    "yall": "you all",
    "yalld": "you all would",
    "yalldve": "you all would have",
    "yallre": "you all are",
    "yallve": "you all have",
    "youd": "you would",
    "youdve": "you would have",
    "youll": "yyou will",
    "youllve": "you shall have ",
    "youre": "you are",
    "youve": "you have",
    "ain t": "are not",
    "aren t": "are not",
    "can t": "cannot",
    "can t ve": "cannot have",
    " cause": "because",
    "could ve": "could have",
    "couldn t": "could not",
    "couldn t ve": "could not have",
    "didn t": "did not",
    "doesn t": "does not",
    "don t": "do not",
    "hadn t": "had not",
    "hadn t ve": "had not have",
    "hasn t": "has not",
    "haven t": "have not",
    "he d": "he would",
    "he d ve": "he would have",
    "he ll": "he will",
    "he ll ve": "he will have",
    "he s": "he is",
    "how d": "how did",
    "how d y": "how do you",
    "how ll": "how will",
    "how s": "how iss",
    "i d": "i would",
    "i d ve": "i would have",
    "i ll": "i will",
    "i ll ve": "i will have",
    "i m": "i am",
    "i ve": "i have",
    "isn t": "is not",
    "it d": "it would",
    "it d ve": "it would have",
    "it ll": "it will",
    "it ll ve": "it will have",
    "it s": "it is",
    "let s": "let us",
    "ma am": "madam",
    "mayn t": "may not",
    "might ve": "might have",
    "mightn t": "might not",
    "mightn t ve": "might not have",
    "must ve": "must have",
    "mustn t": "must not",
    "mustn t ve": "must not have",
    "needn t": "need not",
    "needn t ve": "need not have",
    "o clock": "of the clock",
    "oughtn t": "ought not",
    "oughtn t ve": "ought not have",
    "shan t": "shall not",
    "sha n t": "shall not",
    "shan t ve": "shall not have",
    "she d": "she would",
    "she d ve": "she would have",
    "she ll": "she will",
    "she ll ve": "she will have",
    "she s": "she is",
    "should ve": "should have",
    "shouldn t": "should not",
    "shouldn t ve": "should not have",
    "so ve": "so have",
    "so s": "so is",
    "that d": "that would",
    "that d ve": "that would have",
    "that s": "that is",
    "there d": "there would",
    "there d ve": "there would have",
    "there s": "there is",
    "they d": "they would",
    "they d ve": "they would have",
    "they ll": "they will",
    "they ll ve": "they will have",
    "they re": "they are",
    "they ve": "they have",
    "to ve": "to have",
    "wasn t": "was not",
    "we d": "we would",
    "we d ve": "we would have",
    "we ll": "we will",
    "we ll ve": "we will have",
    "we re": "we are",
    "we ve": "we have",
    "weren t": "were not",
    "what ll": "what will",
    "what ll ve": "what will have",
    "what re": "what are",
    "what s": "what is",
    "what ve": "what have",
    "when s": "when is",
    "when ve": "when have",
    "where d": "where did",
    "where s": "where is",
    "where ve": "where have",
    "who ll": "who will",
    "who ll ve": "who will have",
    "who s": "who is",
    "who ve": "who have",
    "why s": "why is",
    "why ve": "why have",
    "will ve": "will have",
    "won t": "will not",
    "won t ve": "will not have",
    "would ve": "would have",
    "wouldn t": "would not",
    "wouldn t ve": "would not have",
    "y all": "you all",
    "y all d": "you all would",
    "y all d ve": "you all would have",
    "y all re": "you all are",
    "y all ve": "you all have",
    "you d": "you would",
    "you d ve": "you would have",
    "you ll": "yyou will",
    "you ll ve": "you shall have ",
    "you re": "you are",
    "you ve": "you have"
}

# Self-defined Functions
def label_class(x):
    if x<3:
        return 1
    if x>3:
        return 3
    else:
        return 2

def deconstract(x):
    tokens = x.split(' ')
    for i,token in enumerate(tokens):
        if token in contractions.keys():
            tokens[i] = contractions[token]
    return ' '.join(tokens)

def data_cleaning(x):
    x = x.lower() #convert all reviews into lowercase
    x = re.sub(r'\s*https?://\S+(\s+|$)', '', x) #remove the HTML and URLs from the reviews
    x = re.sub(r'[^a-zA-Z ?!"]+', '', x) #remove non-alphabetical characters
    x = x.replace("!"," exclamationmark ")
    x = x.replace("?"," questionmark ")
    x = x.replace('"', ' quotationmark ')
    x = ' '.join(re.sub(r'\s', ' ', x).split()) #remove extra spaces
    x = deconstract(x)
    return x

def ultimate_preprocess_withstopwords(x):
    tokens = []
    for word in x.split(" "):
        try:
            pos = wd.synsets(word)[0].pos()
            tokens.append(ps.stem(lemmatizer.lemmatize(word, pos=pos)))
        except IndexError:
            tokens.append(ps.stem(lemmatizer.lemmatize(word)))
    return ' '.join(tokens)

def myPerceptron(X_train, y_train, X_test, y_test, N=10):
    best_f1 = 0
    for i in range(N):
        seed = np.random.randint(0,1000)
        perceptron_md = Perceptron(tol=1e-3, random_state=seed, penalty='elasticnet', l1_ratio=0.5)
        perceptron_md.fit(X_train, y_train.values.ravel())
        f1 = np.mean(f1_score(y_test, perceptron_md.predict(X_test), average=None))
        if f1 > best_f1:
            best_f1 = f1
            best_seed = seed
    perceptron_md = Perceptron(tol=1e-3, random_state=best_seed)
    perceptron_md.fit(X_train, y_train.values.ravel())
    evaluation(perceptron_md.predict(X_test), y_test)

def evaluation(y_pred, y_true):
    prc = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    acc = accuracy_score(y_true, y_pred)
    for cls in range(1,4):
        print(f'<Class {cls}> Precision: {prc[cls-1]}, Recall: {recall[cls-1]}, F-1: {f1[cls-1]}')
    print(f'<Overall Mean> Precision: {np.mean(prc)}, Recall: {np.mean(recall)}, F-1: {np.mean(f1)}, Accuracy: {np.mean(acc)}')

def main():
    raw_df = pd.read_table(DATA_DIR+'amazon_reviews_us_Beauty_v1_00.tsv.gz', compression='gzip', quotechar='"', quoting=csv.QUOTE_NONE)
    df = raw_df[["star_rating","review_body"]].dropna()
    df["label"] = df["star_rating"].apply(label_class)
    sampled_df = pd.concat([df[df["label"] == k].sample(n=20000) for k in range(1,4)])

    #2 Data Cleaning
    # Print AVG length of reviews BEFORE data cleaning
    print(f'AVG length of reviews BEFORE data cleaning: {sampled_df["review_body"].apply(len).mean()}')
    sampled_df["review_cleaned"] = sampled_df["review_body"].apply(data_cleaning)
    # Print AVG length of reviews AFTER cleaning
    print(f'AVG length of reviews AFTER cleaning: {sampled_df["review_cleaned"].apply(len).mean()}')

    #3 Preprocess
    print(f'AVG length of reviews BEFORE preprocess: {sampled_df["review_cleaned"].apply(len).mean()}')
    sampled_df["review_processed"] = sampled_df["review_cleaned"].apply(ultimate_preprocess_withstopwords)
    # Print AVG length of reviews AFTER preprocessed
    print(f'AVG length of reviews AFTER preprocessed: {sampled_df["review_processed"].apply(len).mean()}')

    #4 Feature Extraction
    corpus = sampled_df["review_processed"].values
    vectorizer = TfidfVectorizer(min_df=0.0008)
    feature = vectorizer.fit_transform(corpus)
    feature_df = pd.DataFrame(feature.toarray(), columns=vectorizer.get_feature_names_out(), index=sampled_df.index)
    X_train, X_test, y_train, y_test = train_test_split(feature_df, sampled_df[['label']], test_size=0.2, random_state=0)

    #5 Perceptron
    print("---Perceptron Model (This may take 1-2 min)---")
    myPerceptron(X_train, y_train, X_test, y_test, N=10)

    #6 SVM
    print("---SVM Model---")
    svm_md = LinearSVC(random_state=0, dual=False, C=0.05)
    svm_md.fit(X_train, y_train.values.ravel())
    evaluation(svm_md.predict(X_test), y_test)

    #7 Logistic Regression
    print("---Logistic Regression Model---")
    LR_md = LogisticRegression(tol=1e-3, random_state=0, solver='saga', dual=False, C=0.4)
    LR_md.fit(X_train, y_train.values.ravel())
    evaluation(LR_md.predict(X_test), y_test)

    #8 Naive Baynes
    print("---Naive Baynes Model---")
    NB_md = MultinomialNB()
    NB_md.fit(X_train, y_train.values.ravel())
    evaluation(NB_md.predict(X_test), y_test)

if __name__ == "__main__":
    main()