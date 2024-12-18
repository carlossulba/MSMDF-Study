# ClassifierTrainer.py

from enum import Enum
import numpy as np
import pandas as pd
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Set, Any
from math import inf
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit, train_test_split, cross_val_predict, cross_validate)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
# import xgboost as xgb
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from matplotlib.colors import ListedColormap


from FingerprintExtractor import FingerprintExtractor, FingerprintConfig, FingerprintSetting, FingerprintSensor, FingerprintDataStream, FingerprintFeature


class Classifiers(Enum):
    """Supported classifiers for training and evaluation."""
    # Linear Models
    # LOGISTIC_REGRESSION = LogisticRegression
    STOCHASTIC_GRADIENT_DESCENT = SGDClassifier
    LINEAR_DISCRIMINANT_ANALYSIS = LinearDiscriminantAnalysis
    
    # Support Vector Machines
    SUPPORT_VECTOR_MACHINE = SVC
    
    # Naive Bayes
    GAUSSIAN_NAIVE_BAYES = GaussianNB
    
    # Nearest Neighbors
    K_NEAREST_NEIGHBORS = KNeighborsClassifier
    BAGGED_KNN = BaggingClassifier # (estimator=KNeighborsClassifier())
    
    # Tree-Based Models
    DECISION_TREE = DecisionTreeClassifier
    
    # Ensemble Methods
    BAGGED_DECISION_TREES = BaggingClassifier # (estimator=DecisionTreeClassifier())
    RANDOM_FOREST = RandomForestClassifier
    EXTRA_TREES = ExtraTreesClassifier
    
    # Gradient Boosting
    # XGBOOST = xgb.XGBClassifier
    
    # Neural Networks
    MULTILAYER_PERCEPTRON = MLPClassifier # (hidden_layer_sizes=(100,100))
    WIDE_NEURAL_NETWORK = MLPClassifier # (hidden_layer_sizes=(100))

class TrainingMethod(Enum):
    """Supported training methods for classifiers."""
    CLASSIC = 'classic'
    GRID_SEARCH = 'grid-search'
    RANDOM_SEARCH = 'random-search'
    
class EvaluationMethod(Enum):
    """Supported evaluation methods for classifiers."""
    CLASSIC = 'classic'
    CROSS_VALIDATION = 'cross-validation'


@dataclass
class EvaluationConfig:
    """Configuration for evaluation of classifiers."""
    num_devices: int = 100
    training_set_ratio: float = 0.8
    known_unknown_ratio: float = 1.0
    
    cv_folds: int = 5
    random_state: int = 42
    
    classifiers: set = field(default_factory=lambda: {c for c in Classifiers})
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.num_devices <= 100:
            raise ValueError("num_devices must be between 0 and 100")
        if not 0 < self.training_set_ratio < 1:
            raise ValueError("training_set_ratio must be between 0 and 1")   
        
        if not 0 < self.known_unknown_ratio <= 1:
            raise ValueError("known_unknown_ratio must be between 0 and 1")
        if not 0 < self.cv_folds <= 10:
            raise ValueError("cv_folds must be be between 1 and 10")
        if self.random_state < 0:
            raise ValueError("random_state must be a non-negative integer")
        
        self._validate_items(self.classifiers, "classifiers", Classifiers)
        
    def _validate_data(self, data: Any):
        """Validate the input data in the configuration."""
        if isinstance(self.data, dict):
            if not all(isinstance(k, str) and isinstance(v, list) for k, v in data.items()):
                raise ValueError("Data dictionary must have string keys and list values.")

        elif isinstance(self.data, str):
            # 'data' is a string; check if it's a file path
            if os.path.isfile(self.data):
                file_extension = os.path.splitext(self.data)[1].lower()
                try:
                    if file_extension == '.pkl':
                        try:
                            with open(self.data, 'rb') as f:
                                loaded_data = pickle.load(f)
                            if not isinstance(loaded_data, dict):
                                raise TypeError(f"Loaded data from '{data}' is not a dictionary.")
                            self.data = loaded_data
                        except Exception as e:
                            print(f"Failed to load data: {e}") 

                    elif file_extension == '.json':
                        with open(self.data, 'r') as f:
                            loaded_data = json.load(f)
                        if not isinstance(loaded_data, dict):
                            raise TypeError(f"Loaded data from '{data}' is not a dictionary.")
                        self.data = loaded_data
                    else:
                        raise ValueError(f"Unsupported file extension '{file_extension}'. Only '.pkl' and '.json' are supported.")
                except Exception as e:
                    raise IOError(f"Failed to load data from file '{self.data}': {e}")
            else:
                raise FileNotFoundError(f"The file '{self.data}' does not exist.")
        else:
            raise TypeError(f"'data' must be a dictionary or a valid file path to a '.pkl' or '.json' file, got {type(self.data).__name__}")
     
    def _validate_items(self, items: set, field_name: str, enum_type: type):
        """Validate that all elements in a set are valid members or values of an Enum."""
        if not all(isinstance(item, enum_type) for item in items):
            invalid_items = [item for item in items if not isinstance(item, enum_type)]
            raise ValueError(f"Invalid values in '{field_name}': {invalid_items}.")
        

class ClassifierTrainer:
    """ Main class for training and evaluating classifiers."""
    
    def __init__(self, config: EvaluationConfig, log_level: str = 'INFO'):
        """
        Initialize the classifier trainer.
        
        Args:
            config: Configuration for training and evaluation
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        self.config = config
        self.dataset_config = None
        self.hyperparameter_grids = {}

        self.trained_classifiers = {}
        self.evaluation_results = {}
        
        self._setup_logger(log_level)
        
    def _setup_logger(self, log_level: str):
        """Setup logging for the classifier trainer."""
        basic_format = '%(levelname)s - %(message)s'
        detailed_format = '%(levelname)s - %(message)s [%(filename)s:%(lineno)d]'
        
        # Use detailed format for levels other than INFO
        log_format = basic_format if log_level.upper() == 'INFO' else detailed_format
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format=log_format
        )
        self.logger = logging.getLogger(__name__)
        
    def load_and_preprocess_data(self, data: Any) -> tuple:
        """
        Load and preprocess data with comprehensive NaN handling.
        
        Args:
            data (Any): Can be one of the following:
                - str: Path to a pickle file containing the fingerprints dictionary.
                - str: Path to a directory containing 'features.npy' and 'labels.npy' files.
                - dict: The fingerprints dictionary already loaded.
            
        Returns:
            tuple: (X_train, X_holdout, y_train, y_holdout, label_encoding)
        """
        try:
            # Initialize features and labels
            features = None
            labels = None
            feature_sources = []
            
            # Case 1 and 2: If data is a string
            if isinstance(data, str):
                if data.endswith(('.pkl', '.pickle')):
                    # Case 1: Load fingerprints dictionary from pickle file
                    with open(data, 'rb') as f:
                        data_loaded = pickle.load(f)
                    # Extract 'fingerprints' and 'config'
                    fingerprints_dict = data_loaded.get('fingerprints')
                    dataset_config = data_loaded.get('config')
                    
                    if fingerprints_dict is None:
                        raise KeyError("The pickle file does not contain 'fingerprints'.")
                    if dataset_config is None:
                        raise KeyError("The pickle file does not contain 'config'.")

                    # Print the configuration
                    self.logger.info("Loaded configuration from pickle file:")
                    self.dataset_config = dataset_config
                    print(dataset_config)
                    
                else:
                    # Case 2: Load features and labels from directory
                    feature_path = os.path.join(data, 'features.npy')
                    label_path = os.path.join(data, 'labels.npy')

                    if not os.path.isfile(feature_path) or not os.path.isfile(label_path):
                        raise FileNotFoundError("Features or labels file not found in the specified directory.")

                    features = np.load(feature_path)
                    labels = np.load(label_path)
                    feature_sources = [{'row_index': idx} for idx in range(len(features))]

            # Case 3: If data is a dictionary
            elif isinstance(data, dict):
                fingerprints_dict = data
            else:
                raise ValueError("Unsupported data type for 'data' parameter.")
            
            # If fingerprints_dict is defined, process it to extract features and labels
            if 'fingerprints_dict' in locals():
                features_list = []
                labels_list = []

                for setting, devices in fingerprints_dict.items():
                    for device_id, recordings in devices.items():
                        for recording_idx, (fingerprints_list_item, _) in enumerate(recordings):
                            for fp_idx, fp in enumerate(fingerprints_list_item):
                                features_list.append(fp.flatten())
                                labels_list.append(device_id)  # set device_id as the label
                                feature_sources.append({
                                    'setting': setting,
                                    'device_id': device_id,
                                    'recording_idx': recording_idx,
                                    'fingerprint_idx': fp_idx
                                })

                features = np.array(features_list)
                labels = np.array(labels_list)

            # Check if features and labels are loaded
            if features is None or labels is None:
                raise ValueError("Features and labels could not be loaded.")
            
            # Check for NaN values
            nan_mask = np.isnan(features)
            total_nans = np.sum(nan_mask)
            
            if total_nans > 0:
                self.logger.warning(f"Total NaN values: {total_nans}")
                
                # Identify rows with NaNs
                rows_with_nans = np.where(nan_mask.any(axis=1))[0]
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    num_rows_with_nans = len(rows_with_nans)
                    self.logger.debug(f"Number of feature vectors containing NaNs: {num_rows_with_nans}")

                    for row in rows_with_nans:
                        nan_cols_in_row = np.where(nan_mask[row])[0]
                        source_info = feature_sources[row]
                        self.logger.debug(
                            f"Feature vector at row {row} "
                            f"(Setting: {source_info.get('setting', 'N/A')}, "
                            f"Device ID: {source_info.get('device_id', 'N/A')}, "
                            f"Recording Index: {source_info.get('recording_idx', 'N/A')}, "
                            f"Fingerprint Index: {source_info.get('fingerprint_idx', 'N/A')}) "
                            f"has NaNs in columns {nan_cols_in_row.tolist()}"
                        )
                    
                # Handle NaN values
                imputer = SimpleImputer(strategy='mean')
                features = imputer.fit_transform(features)
                self.logger.warning(f"Replaced NaN values using SimpleImputer with strategy 'mean'.")
            
            # Verify dimensions match
            if len(features) != len(labels):
                raise AssertionError(f"Feature and label lengths don't match: {len(features)} vs {len(labels)}")
            
            # Encode labels
            encoder = LabelEncoder()
            encoded_labels = encoder.fit_transform(labels)
            label_encoding = {label: idx for idx, label in enumerate(encoder.classes_)}
            
            num_input_devices = len(encoder.classes_)
            
            # Select a subset of devices
            if num_input_devices < self.config.num_devices:
                self.logger.warning(
                    f"Requested number of devices ({self.config.num_devices}) exceeds available devices ({num_input_devices}). "
                    f"Using all available devices."
                )
                self.config.num_devices = num_input_devices
            
            selected_devices = encoder.classes_[:self.config.num_devices]
            selected_indices = [i for i, label in enumerate(labels) if label in selected_devices]
            subset_features = features[selected_indices]
            subset_labels = encoded_labels[selected_indices]
            
            # Split devices into known and unknown
            known_device_count = int(self.config.num_devices * self.config.known_unknown_ratio)
            if known_device_count < 1 or known_device_count > self.config.num_devices:
                raise ValueError(
                    f"Invalid known-unknown ratio: {self.config.known_unknown_ratio}. "
                    f"Must result in at least 1 known device. "
                    f"Current settings yield {known_device_count} known devices out of {self.config.num_devices} selected."
                )

            known_devices = selected_devices[:known_device_count]
            known_device_indices = [encoder.transform([device])[0] for device in known_devices]
            unknown_devices = selected_devices[known_device_count:]
            unknown_device_indices = [encoder.transform([device])[0] for device in unknown_devices]

            known_indices = [i for i, label in enumerate(subset_labels) if label in known_device_indices]
            unknown_indices = [i for i, label in enumerate(subset_labels) if label in unknown_device_indices]

            # Ensure there is data for training and testing
            if len(known_indices) < 2 :
                raise ValueError(
                    f"Insufficient data for training or testing."
                    f"Adjust the number of devices ({num_input_devices} available) or the known-unknown ratio ({known_device_count} current known)."
                    f"Eventually, the dataset ({num_input_devices} devices in the dataset)."
                )

            # Separate known data
            known_features = subset_features[known_indices]
            known_labels = subset_labels[known_indices]

            # Separate unknown data
            unknown_features = subset_features[unknown_indices]
            unknown_labels = subset_labels[unknown_indices]

            # Split the dataset into training and testing set
            X_train, X_test_known, y_train, y_test_known = train_test_split(
                known_features, 
                known_labels, 
                test_size=1-self.config.training_set_ratio, 
                random_state=self.config.random_state,
                stratify=known_labels  # Ensure class proportions are preserved
            )
            
            # Combine known test data with unknown device data
            X_test = np.vstack((X_test_known, unknown_features))
            y_test = np.hstack((y_test_known, unknown_labels))
            
            self.logger.info(f"Training set shape: {X_train.shape}")
            self.logger.info(f"Test set shape: {X_test.shape}")
            self.logger.info(f"Number of classes: {self.config.num_devices}")
            self.logger.info(f"Known devices: {len(known_devices)}, Unknown devices: {len(unknown_devices)}\n")
            
            return X_train, X_test, y_train, y_test, label_encoding, dataset_config
        
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise
    
    def train_classifiers(self, X_train: np.ndarray, y_train: np.ndarray, method:TrainingMethod) -> dict:
        """
        Train specified classifiers with the given method.
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: Training method to use
            scaling: Whether to scale the data before training
            
        Returns:
            dict: A dictionary of trained classifier models.
        """
        self.logger.info(f"\nTraining classifiers using '{method.name.lower()}' method...")
        
        # Initialize the classifiers and hyperparameter grids
        classifiers_to_train = self._initialize_classifiers()
        self.hyperparameter_grids = self._initialize_hyperparameter_grids()    
        
        # Train each classifier
        for classifier_name, classifier_class in classifiers_to_train.items():
            # Create pipeline to avoid data leakage during scaling
            steps = []
            steps.append(('scaler', StandardScaler()))
            steps.append(('classifier', classifier_class))
            pipeline = Pipeline(steps)
        
            # Train the classifier based on the chosen method
            self.logger.info(f"Training {classifier_name} ...")
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                
                    if method == TrainingMethod.CLASSIC:
                        pipeline.fit(X_train, y_train)                
                        self.trained_classifiers[classifier_name] = pipeline
                        
                    elif method == TrainingMethod.GRID_SEARCH:
                        grid_search = GridSearchCV(
                            estimator=pipeline,
                            param_grid=self.hyperparameter_grids.get(classifier_name, {}),
                            scoring='accuracy',
                            n_jobs=-1,
                            verbose=0,
                            cv=StratifiedKFold(n_splits=self.config.cv_folds)
                        )
                        grid_search.fit(X_train, y_train)
                        self.trained_classifiers[classifier_name] = grid_search.best_estimator_
                        
                    elif method == TrainingMethod.RANDOM_SEARCH:                                        
                        random_search  = RandomizedSearchCV(
                            estimator=pipeline,
                            param_distributions=self.hyperparameter_grids.get(classifier_name, {}),
                            n_iter=10,
                            scoring='accuracy',
                            n_jobs=-1,
                            verbose=0,
                            cv=StratifiedKFold(n_splits=self.config.cv_folds),
                            random_state=self.config.random_state
                        )
                        random_search.fit(X_train, y_train)
                        self.trained_classifiers[classifier_name] = random_search.best_estimator_
                    
                    else:
                        raise ValueError(f"Invalid training method: ({method}). Valid options are {set(TrainingMethod)}")

                    # Check for convergence warnings
                    convergence_warnings = [
                        warning for warning in caught_warnings 
                        if issubclass(warning.category, ConvergenceWarning)
                    ]
                    
                    if convergence_warnings:
                        self.logger.warning(f"  {classifier_name} did not converge. Warnings: {[str(w.message) for w in convergence_warnings]}")
                        continue

            except Exception as e:
                self.logger.error(f"Failed to train {classifier_name}: {e}")
                continue
            
            self.logger.info(f"  Trained {classifier_name} successfully.")
            
        return self.trained_classifiers
    
    def _initialize_classifiers(self):
        """Initialize all classifier instances."""
        initialized_classifiers = {}
        
        for classifier_enum in self.config.classifiers:
            classifier_name = classifier_enum.name.lower()
            classifier_class = classifier_enum.value
            classifier = None
            
            try:
                if classifier_name == 'logistic_regression':
                    classifier = classifier_class(
                        multi_class='multinomial',
                        random_state=self.config.random_state
                    )
                elif classifier_name == 'stochastic_gradient_descent':
                    classifier = classifier_class(
                        random_state=self.config.random_state
                    )
                elif classifier_name == 'linear_discriminant_analysis':
                    classifier = classifier_class(
                    )
                elif classifier_name == 'support_vector_machine':
                    classifier = classifier_class(
                        random_state=self.config.random_state
                    )
                elif classifier_name == 'gaussian_naive_bayes':
                    classifier = classifier_class(
                    )
                elif classifier_name == 'k_nearest_neighbors':
                    classifier = classifier_class(
                    )
                elif classifier_name == 'bagged_knn':
                    classifier = classifier_class(
                        estimator=KNeighborsClassifier(),
                        random_state=self.config.random_state
                    )
                elif classifier_name == 'decision_tree':
                    classifier = classifier_class(
                        random_state=self.config.random_state
                    )
                elif classifier_name == 'bagged_decision_trees':
                    classifier = classifier_class(
                        estimator=DecisionTreeClassifier(),
                        random_state=self.config.random_state
                    )
                elif classifier_name == 'random_forest':
                    classifier = classifier_class(
                        random_state=self.config.random_state
                    )
                elif classifier_name == 'extra_trees':
                    classifier = classifier_class(
                        random_state=self.config.random_state
                    )
                elif classifier_name == 'xgboost':
                    classifier = classifier_class(
                        num_class=self.config.num_devices,
                        random_state=self.config.random_state
                    )
                elif classifier_name == 'multilayer_perceptron':
                    classifier = classifier_class(
                        hidden_layer_sizes=(100,100),
                        random_state=self.config.random_state
                    )
                elif classifier_name == 'wide_neural_network':
                    classifier = classifier_class(
                        hidden_layer_sizes=(100,),
                        random_state=self.config.random_state
                    )   
                else:
                    raise ValueError(f"Unsupported classifier: {classifier_name}")                 
            
            except Exception as e:
                raise Exception(f"Failed to initialize classifier '{classifier_name}': {e}")
            
            initialized_classifiers[classifier_name] = classifier

        return initialized_classifiers

    def _initialize_hyperparameter_grids(self):
        """Initialize hyperparameter grids for grid search."""
        
        hyperparameter_grids = {
            # Linear Models
            'logistic_regression': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['lbfgs', 'saga'],
                'classifier__max_iter': [100, 200, 500]
            },
            'stochastic_gradient_descent': {
                'classifier__loss': ['hinge', 'log_loss', 'modified_huber'],
                'classifier__alpha': [0.0001, 0.001],
                'classifier__max_iter': [1000],
            },
            'linear_discriminant_analysis': [
                {'classifier__solver': ['svd']},
                {'classifier__solver': ['lsqr'], 'classifier__shrinkage': [None, 'auto']}
            ],
            
            # Support Vector Machines
            'support_vector_machine': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__gamma': ['scale', 'auto']
            },
            
            # Naive Bayes
            'gaussian_naive_bayes': {
                'classifier__var_smoothing': [1e-9, 1e-7]
            },
            
            # Nearest Neighbors
            'k_nearest_neighbors': {
                'classifier__n_neighbors': [3, 5, 7],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['minkowski', 'manhattan', 'euclidean']
            },
            'bagged_knn': {
                'classifier__n_estimators': [10, 20],
                'classifier__max_samples': [0.5, 1.0],
                'classifier__estimator__n_neighbors': [3, 5],
                'classifier__estimator__weights': ['uniform', 'distance']
            },
            
            # Tree-Based Models 
            'decision_tree': {
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5]
            },
            
            # Ensemble Methods
            'bagged_decision_trees': {
                'classifier__n_estimators': [10, 20, 50, 100],
                'classifier__max_samples': [0.5, 1.0],
                'classifier__estimator__max_depth': [5, 10, None]
            },
            'random_forest': {
                'classifier__n_estimators': [10, 20, 50, 100],
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5]
            },
            'extra_trees': {
                'classifier__n_estimators': [10, 20, 50, 100],
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5]
            },
            
            # Gradient Boosting
            'xgboost': {
                'classifier__num_class': self.config.num_devices,
                'classifier__n_estimators': [3, 5, 10, 50, 100],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5, 7],
                'classifier__objective': ['multi:softmax', 'multi:softprob']
            },
            
            # Neural Networks
            'multilayer_perceptron': {
                'classifier__hidden_layer_sizes': [(10,10), (20,20), (50,50), (100,100)],
                'classifier__solver': ['adam', 'sgd'],
                'classifier__alpha': [0.0001, 0.001, 0.01],
                'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
                'classifier__early_stopping': [True],
                'classifier__max_iter': [1000]
            },
            'wide_neural_network': {
                'classifier__hidden_layer_sizes': [(25), (50), (100), (200), (400)],
                'classifier__solver': ['adam', 'sgd'],
                'classifier__alpha': [0.0001, 0.001, 0.01],
                'classifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
                'classifier__early_stopping': [True],
                'classifier__max_iter': [1000]
            }
        }
    
        return hyperparameter_grids

    def evaluate_classifiers(self, X_test: np.ndarray, y_test: np.ndarray, method: TrainingMethod, classifiers: dict = None) -> dict:
        """
        Evaluate specified classifiers with the given method.
    
        Args:
            X_test: Features for evaluation
            y_test: Labels for evaluation
            method: Evaluation method to use
            classifiers: Dictionary of classifiers to evaluate (default: trained classifiers)
            
        Returns:
            dict: A dictionary of evaluation results for the classifiers.
        """
        self.logger.info(f"\nEvaluating classifiers using '{method.name.lower()}' method...")
        
        # Use the specified classifiers or default to trained classifiers
        classifiers_to_evaluate = classifiers or self.trained_classifiers
        if not classifiers_to_evaluate:
            raise ValueError("No classifiers to evaluate. Please train or provide classifiers.")
        
        # Evaluate each classifier
        for classifier_name, pipeline in classifiers_to_evaluate.items():
            
            # Evaluate the classifier based on the chosen method
            self.logger.info(f"Evaluating {classifier_name} ...")
            try:
                if method == EvaluationMethod.CLASSIC:
                    y_pred = pipeline.predict(X_test)
                    self.evaluation_results[classifier_name] = self._compute_metrics(y_test, y_pred)
                    
                elif method == EvaluationMethod.CROSS_VALIDATION:
                    cv = StratifiedKFold(
                        n_splits=self.config.cv_folds, 
                        shuffle=True, 
                        random_state=self.config.random_state
                    )
                    y_pred = cross_val_predict(
                        estimator=pipeline, 
                        X=X_test, 
                        y=y_test, 
                        cv=cv
                    )
                    self.evaluation_results[classifier_name] = self._compute_metrics(y_test, y_pred)
                
                else:
                    raise ValueError(f"Invalid evaluation method: ({method}). Valid options are {set(EvaluationMethod)}")
            
            except Exception as e:
                self.logger.error(f"Failed to evaluate {classifier_name}: {e}")
                continue
            
            self.logger.info(f"Evaluated {classifier_name} successfully.")
        
        return self.evaluation_results
                
    def _compute_metrics(self, y_true, y_pred) -> dict:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            dict: A dictionary containing overall, per-class metrics and confusion matrix.
        """
        model_stats = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }

        # Get per-class statistics
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Create confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        return {
            'Overall Metrics': model_stats,
            'Per Class Metrics': class_report,
            'Confusion Matrix': conf_matrix
        }

    def print_evaluation_summary(self, results: dict = None, print_per_class: bool = False, print_conf_matrix: bool = False)  -> None:
        """Print a summary of evaluation results."""
        
        # Use the specified classifiers or default to trained classifiers
        results_to_print = results or self.evaluation_results
        if not results_to_print:
            raise ValueError("No evaluation results to print. Please evaluate or provide results.")

        # Print the summary
        print("\nEvaluation Summary:")
        print("=" * 50)
        num_classifiers = len(results_to_print)
        print(f"Number of Classifiers: {num_classifiers}")
        num_classes = self.config.num_devices
        print(f"Number of Classes: {num_classes}")
        print("=" * 50)
        
        # Print evaluation results for each classifier
        for classifier_name, performance_metrics in results_to_print.items():
            print(f"\nClassifier: {classifier_name}")
            print("-" * 50)            
            
            print("Overall Metrics:")
            overall_metrics = performance_metrics.get('Overall Metrics', {})
            for metric_name, metric_value in overall_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
            
            if print_per_class == True:
                print("Per-Class Metrics:")
                per_class_metrics = performance_metrics.get('Per Class Metrics', {})
                for class_label, class_metrics in per_class_metrics.items():
                    if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                        print(f"  Class {class_label}:")
                        print(f"    Precision: {class_metrics['precision']:.4f}")
                        print(f"    Recall: {class_metrics['recall']:.4f}")
                        print(f"    F1 Score: {class_metrics['f1-score']:.4f}")
                        print(f"    Support: {class_metrics['support']}")
            
            if print_conf_matrix == True:
                print("Confusion Matrix:")
                conf_matrix = performance_metrics.get('Confusion Matrix', [])
                # Prepare column and row labels
                header = [" "] + [f"Predicted {i}" for i in range(len(conf_matrix))]
                rows = [f"Actual {i}" for i in range(len(conf_matrix))]
                
                # Format the matrix into rows
                formatted_rows = []
                for row_label, row_values in zip(rows, conf_matrix):
                    formatted_rows.append(f"{row_label:10} " + " ".join(f"{value:8}" for value in row_values))
                
                # Combine header and rows
                header_str = " ".join(f"{col:8}" for col in header)
                matrix_str = "\n".join(formatted_rows)
                
                print(f"  {header_str}\n  {matrix_str}")
            
            print("-" * 50)

    def plot_evaluation_summary(self, results: dict = None, eval_config : EvaluationConfig = None, dataset_config: FingerprintConfig = None, save_directory: str = None) -> None:
        """ Plot evaluation results. """
        results_to_print = results or self.evaluation_results
        evaluation_config = eval_config  or self.config
        
        if not results_to_print:
            raise ValueError("No evaluation results to plot. Please evaluate or provide results.")

        # Define colors of heatmap
        custom_colors = ["#F7FFE8", "#FFC300", "#FF5733", "#C70039", "#900C3F", "#581845"]
        custom_cmap = ListedColormap(custom_colors)

        # Create a separate plot for each classifier
        for classifier_name, performance_metrics in results_to_print.items():
            # Setup figure
            fig = plt.figure(figsize=(15, 12), constrained_layout=True)
            gs = fig.add_gridspec(nrows=3, ncols=2, 
                                height_ratios=[2, 4, 4], 
                                width_ratios=[1, 1])
            
            # Display Configurations
            config_ax = fig.add_subplot(gs[0, :])
            
            eval_config_text = (
                f"Evaluation Configuration\n"
                f"Number of Devices: {evaluation_config .num_devices} | "
                f"Training Set Ratio: {evaluation_config .training_set_ratio} | "
                f"Known-Unknown Ratio: {evaluation_config .known_unknown_ratio} | "
                f"Cross-Validation Folds: {evaluation_config .cv_folds} | "
                f"Random State: {evaluation_config .random_state}"
            )
            if dataset_config:
                dataset_config_text = (
                    f"\nDataset Configuration:\n"
                    f"Fingerprint Length: {dataset_config.fingerprint_length} | "
                    f"Sampling Rate: {dataset_config.sampling_rate}\n"
                    f"Enabled Settings: {', '.join(map(str, dataset_config.enabled_settings))}\n"
                    f"Enabled Sensors: {', '.join(map(str, dataset_config.enabled_sensors))}\n"
                    f"Enabled Streams: {', '.join(map(str, dataset_config.enabled_streams))}\n"
                    f"Enabled Features: {', '.join(map(str, dataset_config.enabled_features))}"
                )
            else:
                dataset_config_text = "\nNo Dataset Configuration Available"
            
            full_config_text = eval_config_text + dataset_config_text
            
            config_ax.text(0.5, 0.5, full_config_text, 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        fontsize=8, 
                        transform=config_ax.transAxes)
            config_ax.axis('off')
            
            # Overall Metrics Bar Plot
            metrics_ax = fig.add_subplot(gs[1, 0])
            overall_metrics = performance_metrics.get('Overall Metrics', {})
            metrics_names = list(overall_metrics.keys())
            metrics_values = list(overall_metrics.values())
            
            bars = metrics_ax.bar(metrics_names, metrics_values, color="#C70039")
            metrics_ax.set_title(f'{classifier_name} - Overall Metrics')
            metrics_ax.set_ylabel('Score')
            metrics_ax.set_ylim(0.00, 1.00)
            metrics_ax.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                metrics_ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.2f}',
                                ha='center', va='bottom')
            
            # Confusion Matrix Heatmap
            conf_matrix_ax = fig.add_subplot(gs[1:, 1])
            conf_matrix = performance_metrics.get('Confusion Matrix', [])
            sns.heatmap(conf_matrix, 
                        annot=True, 
                        fmt='d', 
                        cmap=custom_cmap, 
                        ax=conf_matrix_ax, 
                        cbar=True)
            conf_matrix_ax.set_title(f'{classifier_name} - Confusion Matrix')
            conf_matrix_ax.set_xlabel('Predicted Label')
            conf_matrix_ax.set_ylabel('Actual Label')
            
            # Set an overall title
            fig.suptitle(f'Evaluation Results for {classifier_name}', fontsize=16)
            
        
            # Save the plot
            if save_directory:
                # Ensure the directory exists
                os.makedirs(save_directory, exist_ok=True)
                
                # Generate a unique filename
                current_date = datetime.now().strftime('%Y-%m-%d__%H-%M')
                filename = f"classifier_evaluation_{classifier_name}_{current_date}.png"
                filepath = os.path.join(save_directory, filename)
                
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Plot saved to: {filepath}")
        
        plt.show()
        
    def save_trained_classifiers(self, trained_classifiers: dict = None, classification_config: EvaluationConfig = None, dataset_config: FingerprintConfig = None, save_directory: str = None, format: str = 'pickle') -> None:
        """ Save the trained classifiers. """
        if trained_classifiers is None:
            trained_classifiers = self.trained_classifiers
        if classification_config is None:
            classification_config = self.config
        if dataset_config is None:
            dataset_config = self.dataset_config
        if save_directory is None or save_directory == "":
            raise ValueError("No save_directory given")
        
        os.makedirs(save_directory, exist_ok=True)
        
        current_date = datetime.now().strftime('%Y-%m-%d__%H-%M')
        
        data_to_save = {
            'classifiers': trained_classifiers,
            'classification_config': classification_config,
            'dataset_config': dataset_config
        }
        
        if format == 'pickle':
            file_path = os.path.join(save_directory, f'trained_classifiers_{current_date}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            self.logger.info(f"Trained classifiers saved to {file_path} in pickle format.")
        else:
            self.logger.error(f"Unsupported format '{format}'. Supported formats are 'pickle'.")


# local main
if __name__ == "__main__":
    # Configure classifier training and evaluation
    classification_config = EvaluationConfig(        
        num_devices=100,
        training_set_ratio=0.8,
        known_unknown_ratio=1.0,
        
        cv_folds=2,
        random_state=42,
        
        classifiers=set(Classifiers)
    )
    
    # Initialize the trainer
    trainer = ClassifierTrainer(classification_config, log_level='INFO')

    # Load and preprocess the data
    X_train, X_test, y_train, y_test, label_encoding, dataset_config = trainer.load_and_preprocess_data(
        data="../Data/extracted fingerprints/extracted_fingerprints_2024-12-17__16-36.pkl"
    )

    # Train classifiers
    trained_classifiers = trainer.train_classifiers(X_train, y_train, method=TrainingMethod.CLASSIC)
    trainer.save_trained_classifiers(trained_classifiers, classification_config, dataset_config, "../Results/ClassifierTrainer/Classifiers/")
    print("Training completed successfully.\n")
    
    # Evaluate classifiers
    evaluation_results = trainer.evaluate_classifiers(X_test, y_test, method=EvaluationMethod.CLASSIC)
    
    # Plot evaluation summary
    trainer.plot_evaluation_summary(evaluation_results, classification_config, dataset_config, "../Results/ClassifierTrainer/Plots/")
    print("Evaluation completed successfully.\n")
    
    
    