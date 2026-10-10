"""Register a model factory here; training and backtesting do not name estimators."""
from dataclasses import dataclass
import inspect
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class TemporalSVM(ClassifierMixin, BaseEstimator):
    """Fit SVC on the prefix and probability calibration on the later training tail."""
    def __init__(self, calibration_fraction=0.2, **params):
        self.calibration_fraction = calibration_fraction
        self.params = params

    def fit(self, X, y):
        if not 0 < self.calibration_fraction < 1:
            raise ValueError('SVM calibration_fraction must be between 0 and 1')
        split = int(len(y) * (1 - self.calibration_fraction))
        y = np.asarray(y)
        if min(split, len(y)-split) < 2 or any(len(np.unique(part)) != 2 for part in (y[:split], y[split:])):
            raise ValueError('SVM needs both classes in the chronological fit/calibration partitions')
        params = dict(self.params)
        if params.pop('probability', False):
            raise ValueError('Use temporal calibration, not SVC probability=True')
        self.estimator_ = make_pipeline(StandardScaler(), SVC(**params))
        self.estimator_.fit(X.iloc[:split], y[:split])
        decision = self.estimator_.decision_function(X.iloc[split:]).reshape(-1, 1)
        self.calibrator_ = LogisticRegression(random_state=42).fit(decision, y[split:])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.calibrator_.predict_proba(self.estimator_.decision_function(X).reshape(-1, 1))


def logistic(params):
    return make_pipeline(StandardScaler(), LogisticRegression(**params))


def knn(params):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(**params))


def logistic_pca(params):
    params = dict(params)
    components = params.pop('pca_n_components', 0.95)
    return make_pipeline(StandardScaler(), PCA(n_components=components, svd_solver='full'),
                         LogisticRegression(**params))


def knn_pca(params):
    params = dict(params)
    components = params.pop('pca_n_components', 0.95)
    return make_pipeline(StandardScaler(), PCA(n_components=components, svd_solver='full'),
                         KNeighborsClassifier(**params))


def rf(params):
    return make_pipeline(StandardScaler(), RandomForestClassifier(**{'random_state': 42, **params}))


def svm(params):
    return TemporalSVM(**params)


def fnn(params):
    if params.get('early_stopping', False):
        raise ValueError('Random early-stopping splits are disabled; temporal early stopping is not implemented')
    return make_pipeline(StandardScaler(), MLPClassifier(**{'random_state': 42, **params}))


@dataclass(frozen=True)
class ModelDefinition:
    factory: object
    version: str = '1'


MODEL_REGISTRY = {name: ModelDefinition(factory) for name, factory in
                  [('logistic', logistic), ('knn', knn), ('rf', rf), ('svm', svm), ('fnn', fnn),
                   ('logistic_pca', logistic_pca), ('knn_pca', knn_pca)]}


def model_signature(name, params):
    if name not in MODEL_REGISTRY or name in ('zero', 'majority', 'buy_hold', 'cash'):
        raise ValueError(f'Unknown or reserved model name: {name}')
    definition = MODEL_REGISTRY[name]
    result = dict(name=name, params=params, version=definition.version,
                  factory=inspect.getsource(definition.factory))
    if name == 'svm':
        result['adapter'] = inspect.getsource(TemporalSVM)
    return result


def build_model(name, params):
    model_signature(name, params)
    return MODEL_REGISTRY[name].factory(params)
