
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from datasets import load_dataset

import evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import load_dataset, Dataset
import numpy as np
from scipy.special import softmax

import pandas as pd
from sklearn.metrics import classification_report


LEARNING_RATE = 5e-5
EPOCHS=5
MAX_SEQUENCE_LENGTH=128
SEED=42

def finetune_sentiment_classifier(model_name, model_path, lang, data_dir):
    output_dir = f"models/classifier/{lang}/{model_name}"
    data_dir = f"data/raw/train/splitted-train-dev-test/{lang}"
    training_args = TrainingArguments(output_dir=output_dir,
                                      overwrite_output_dir=True,
                                     do_train=True,
                                     do_eval=True,
                                     do_predict=False,
                                     learning_rate=LEARNING_RATE,
                                     num_train_epochs=EPOCHS,
                                     save_steps=-1,
                                     per_device_train_batch_size = 8)

    # Set seed before initializing model.
    set_seed(SEED)


    df = pd.read_csv(data_dir + '/train.tsv', sep='\t')
    df = df.dropna()
    train_dataset = Dataset.from_pandas(df)
    label_list = df['label'].unique().tolist()

    df = pd.read_csv(data_dir+ '/dev.tsv', sep='\t')
    df = df.dropna()
    eval_dataset = Dataset.from_pandas(df)
    label_list = df['label'].unique().tolist()

    df = pd.read_csv(data_dir + '/test.tsv', sep='\t')
    df = df.dropna()
    predict_dataset = Dataset.from_pandas(df)
    label_list = df['label'].unique().tolist()

    # Labels
    num_labels = len(label_list)
    print(label_list)

    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    )

    padding = "max_length"


    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    def preprocess_function(examples):
        texts =(examples['text'],)
        result =  tokenizer(*texts, padding=padding, max_length=MAX_SEQUENCE_LENGTH, truncation=True)
        result["label"] = [(int(label_to_id[l]) if l != -1 else -1) for l in examples["label"]]
        return result

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )

    # Get the metric function
    metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def predict_sentiment(model, tokenizer, text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input) 
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    id2label = {0:"positive", 1:"neutral", 2:"negative"}

    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    predicted_label = id2label[ranking[0]]
    return predicted_label

# evaluation method

def evaluate_bert_model(model_name, model_path, lang, file_path):
    print(f"Evaluating: {model_name} for language {lang}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_and_labels = pd.read_csv(file_path, sep='\t') 
    data = data_and_labels.text
    trues = data_and_labels.label 
    preds = []
    l = list(trues)
    for review in data: 
        predicted_label = predict_sentiment(model, tokenizer, review)
        preds.append(predicted_label)


    return classification_report(l, preds) 