import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
from tqdm import tqdm

VOCAB_SIZE = 15000

def get_data_and_labels(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    data = df['text'].to_list()
    labels = df['label'].to_list()
    labels = [t.strip().upper() for t in labels]
    return data, np.array(labels)

def identity_tokenizer(text):
    return text

def get_features(corpus, vocab_size):
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_features=vocab_size,
        tokenizer=identity_tokenizer, # already receiving tokenized text from AUtotokenizer
        lowercase=False,
        token_pattern=None
    )
    vectorizer.fit(corpus)
    X = vectorizer.transform(corpus)
    return X, vectorizer

def train_NB(train_data_path):
    data, labels = get_data_and_labels(train_data_path)
    tokenizer = AutoTokenizer.from_pretrained('Davlan/afro-xlmr-mini')
    tokenized_texts_str = [tokenizer.convert_ids_to_tokens(text) for text in tokenizer(data)['input_ids']]
    features, vectorizer = get_features(tokenized_texts_str, VOCAB_SIZE)
    classifier = MultinomialNB()
    classifier.fit(features, labels)
    # print(classification_report(labels,classifier.predict(features)))
    return classifier, vectorizer    

def evaluate_NB(classifier, vectorizer, test_data_path):
    test_data, test_labels = get_data_and_labels(test_data_path)
    tokenizer = AutoTokenizer.from_pretrained('Davlan/afro-xlmr-mini')
    tokenized_texts_str = [tokenizer.convert_ids_to_tokens(text) for text in tokenizer(test_data)['input_ids']]
    features = vectorizer.transform(tokenized_texts_str)
    pred_labels = classifier.predict(features)
    return classification_report(test_labels, pred_labels)