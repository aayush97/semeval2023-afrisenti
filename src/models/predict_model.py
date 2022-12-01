from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax

import pandas as pd
from sklearn.metrics import classification_report
from pathlib import Path
import click
from src.models.linear_svc import evaluate_SVM
from src.models.naive_bayes import evaluate_NB
from src.models.bert_model import evaluate_bert_model
from src.models.utility import load_classifier_vectorizer



@click.command()
@click.option("--lang", type=click.Choice(["am", "dz", "ha", "ig", "ma", "pcm", "pt", "sw", "yo"], case_sensitive=False))
@click.option("--model", type = click.Choice(["LinearSVM", "NaiveBayes", "naija-roberta-large", "xlm-roberta-small"]))
@click.option("--finetune_classifier", is_flag=True, default=False, help="Finetune classification layer")
def main(lang, model, finetune_classifier):
    print(f"Model: {model}, Language: {lang}")
    data_dir = Path(f'data/raw/train/splitted-train-dev-test/{lang}')
    test_tsv = Path(data_dir, 'test.tsv')
    model_dir = Path(f"models/classifier/{lang}/{model}")
    if model=="LinearSVM":
        classifier, vectorizer = load_classifier_vectorizer(model_dir)
        print(evaluate_SVM(classifier, vectorizer, test_tsv))
    if model=="NaiveBayes":
        classifier, vectorizer = load_classifier_vectorizer(model_dir)
        print(evaluate_NB(classifier, vectorizer, test_tsv))
    if model=="naija-roberta-large":
        model_path = 'Davlan/naija-twitter-sentiment-afriberta-large'
        if finetune_classifier:
            print(evaluate_bert_model(model, model_dir, lang, test_tsv))
        else:
            print(evaluate_bert_model(model, model_path, lang, test_tsv))
    if model=="xlm-roberta-small":
        model_path = 'Davlan/afro-xlmr-small'
        if finetune_classifier:
            print(evaluate_bert_model(model, model_dir, lang, test_tsv))
        else:
            print(evaluate_bert_model(model, model_path, lang, test_tsv))



if __name__=="__main__":
    main()