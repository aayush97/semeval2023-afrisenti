# semeval2023-afrisenti
A low-resource sentiment analysis project for African Languages

### Running the code
1. Clone the repo and `cd` to project directory
2. Install python -- version 3.8.12
2. Run the command `python -m venv .venv` to setup the virtual environment
3. Activate the virtual environment using `source .venv/bin/activate` 
4. Install the requirements `pip install -r requirements.txt`

### Training the model

The following commandline options are available for training the model

```
Usage: python -m src.models.train_model [OPTIONS]

Options:
  --lang [am|dz|ha|ig|ma|pcm|pt|sw|yo]
  --model [LinearSVM|NaiveBayes|naija-roberta-large|xlm-roberta-small]
  --finetune_lm                   Finetune the language model as well
  --finetune_classifier           Finetune classification layer
  --help                          Show this message and exit.
  ```
The `--model` option indicates which model to train.

The `--finetune_classifier` would fine tune the pretrained model on the training data. It is only applicable is the model is either `naija-roberta-large` or `xlm-roberta-small`. When `LinearSVC` or `NaiveBayes` is selected this option is ignored.

The `--finetune_lm` option will finetune the masked language model objective with the traning data for that particular model. It is always used with `--finetune_classifier`.

Example Usage: `python -m src.models.train_model --lang="pcm"  --model="naija-roberta-large" --finetune_classifier --finetune_lm`

### Evaluating the model

The following commandline options are available for evaluating the model


```
Usage: python -m src.models.predict_model [OPTIONS]

Options:
  --lang [am|dz|ha|ig|ma|pcm|pt|sw|yo]
  --model [LinearSVM|NaiveBayes|naija-roberta-large|xlm-roberta-small]
  --finetune_classifier           Use finetuned classification layer
  --help                          Show this message and exit.
```

The `--model` option indicates which model to train.

The `--finetune_classifier` will use the model finetuned on the training data. It is only applicable is the model is either `naija-roberta-large` or `xlm-roberta-small`. When `LinearSVC` or `NaiveBayes` is selected this option is ignored.

Example Usage: `python -m src.models.predict_model --lang="pcm"  --model="naija-roberta-large" --finetune_classifier`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
