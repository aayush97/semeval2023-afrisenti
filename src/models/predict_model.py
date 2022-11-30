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
from src.models.utility import load_classifier_vectorizer



@click.command()
@click.option("--lang", type=click.Choice(["am", "dz", "ha", "ig", "ma", "pcm", "pt", "sw", "yo"], case_sensitive=False))
@click.option("--model", type = click.Choice(["LinearSVM", "NaiveBayes", "naija-roberta-large", "xlm-roberta-small"]))
@click.option("--finetune_lm", is_flag=True, default=False, help="Finetune the language model before running predictions")
@click.option("--finetune_classifier", is_flag=True, default=True, help="Finetune classification layer")
def main(lang, model, finetune_lm, finetune_classifier):
    print(f"Model: {model}, Language: {lang}")
    test_tsv = Path(f'data/raw/train/splitted-train-dev-test/{lang}/test.tsv')
    model_dir = Path(f"models/classifier/{lang}/{model}")
    if model=="LinearSVM":
        classifier, vectorizer = load_classifier_vectorizer(model_dir)
        print(evaluate_SVM(classifier, vectorizer, test_tsv))
    if model=="NaiveBayes":
        classifier, vectorizer = load_classifier_vectorizer(model_dir)
        print(evaluate_NB(classifier, vectorizer, test_tsv))



if __name__=="__main__":
    main()