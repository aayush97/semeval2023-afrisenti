from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax

import pandas as pd
from sklearn.metrics import classification_report
from pathlib import Path
import click
from src.models.linear_svc import train_SVM
from src.models.naive_bayes import train_NB
from src.models.utility import save_classifier_vectorizer



@click.command()
@click.option("--lang", type=click.Choice(["am", "dz", "ha", "ig", "ma", "pcm", "pt", "sw", "yo"], case_sensitive=False))
@click.option("--model", type = click.Choice(["LinearSVM", "NaiveBayes", "naija-roberta-large", "xlm-roberta-small"]))
@click.option("--finetune_lm", is_flag=True, default=False, help="Finetune the language model before running predictions")
@click.option("--finetune_classifier", is_flag=True, default=True, help="Finetune classification layer")
def main(lang, model, finetune_lm, finetune_classifier):
    train_tsv = Path(f'data/raw/train/splitted-train-dev-test/{lang}/train.tsv')
    dev_tsv = Path(f'data/raw/train/splitted-train-dev-test/{lang}/dev.tsv')
    if model=="LinearSVM":
        classifier, vectorizer = train_SVM(train_tsv)
        save_classifier_vectorizer(lang, model, classifier, vectorizer)
    if model=="NaiveBayes":
        classifier, vectorizer = train_NB(train_tsv)
        save_classifier_vectorizer(lang, model, classifier, vectorizer)



if __name__=="__main__":
    main()