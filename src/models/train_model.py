from pathlib import Path
import click
from src.models.linear_svc import train_SVM
from src.models.naive_bayes import train_NB
from src.models.utility import save_classifier_vectorizer
from src.models.bert_model import finetune_sentiment_classifier



@click.command()
@click.option("--lang", type=click.Choice(["am", "dz", "ha", "ig", "ma", "pcm", "pt", "sw", "yo"], case_sensitive=False))
@click.option("--model", type = click.Choice(["LinearSVM", "NaiveBayes", "naija-roberta-large", "xlm-roberta-small"]))
@click.option("--finetune_lm", is_flag=True, default=False, help="Finetune the language model as well")
@click.option("--finetune_classifier", is_flag=True, default=False, help="Finetune classification layer")
def main(lang, model, finetune_lm, finetune_classifier):
    data_dir = Path(f'data/raw/train/splitted-train-dev-test/{lang}')
    train_tsv = Path(data_dir, 'train.tsv')
    if model=="LinearSVM":
        classifier, vectorizer = train_SVM(train_tsv)
        save_classifier_vectorizer(lang, model, classifier, vectorizer)
    if model=="NaiveBayes":
        classifier, vectorizer = train_NB(train_tsv)
        save_classifier_vectorizer(lang, model, classifier, vectorizer)

    if model=="naija-roberta-large":
        model_path = 'Davlan/naija-twitter-sentiment-afriberta-large'
        if finetune_lm:
            model_path = None
        if finetune_classifier:
            finetune_sentiment_classifier(model, model_path, lang, data_dir)
    if model=="xlm-roberta-small":
        model_path = 'Davlan/afro-xlmr-small'
        if finetune_lm:
            model_path = None
        if finetune_classifier:
            finetune_sentiment_classifier(model, model_path, lang, data_dir)
        

if __name__=="__main__":
    main()