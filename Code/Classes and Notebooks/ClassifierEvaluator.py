# ClassifierEvaluator.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassifierEvaluator:
    def __init__(self, classifiers, evaluation_method='default'):
        self.classifiers = classifiers
        self.evaluation_method = evaluation_method

    def evaluate_classifier(self, features, labels):
        performance = {}
        for classifier_name, classifier in self.classifiers.items():
            predictions = classifier.predict(features)
            performance[classifier_name] = {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions, average='weighted'),
                'recall': recall_score(labels, predictions, average='weighted'),
                'f1_score': f1_score(labels, predictions, average='weighted')
            }
        return performance
