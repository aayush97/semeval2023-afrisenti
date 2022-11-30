import pickle
from pathlib import Path


def save_classifier_vectorizer(language, model_name, classifier, vectorizer):
    model_dir = Path(f"models/classifier/{language}/{model_name}")
    model_dir.mkdir(exist_ok=True, parents=True)
    with open(model_dir/"classifier.pickle", "wb") as clf_fh:
        s = pickle.dumps(classifier)
        clf_fh.write(s)
    with open(model_dir/"vectorizer.pickle", "wb") as vec_fh:
        s = pickle.dumps(vectorizer)
        vec_fh.write(s)
    return True

def load_classifier_vectorizer(model_dir):
    with open(Path(model_dir, "classifier.pickle"), "rb") as clf_fh:
        clf = pickle.loads(clf_fh.read())
    with open(Path(model_dir, "vectorizer.pickle"), "rb") as vec_fh:
        vec = pickle.loads(vec_fh.read())
    return clf, vec
    
