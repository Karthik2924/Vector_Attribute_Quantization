import numpy as np
from sklearn import preprocessing, feature_selection, metrics, linear_model


import torch
import torch.nn as nn
from sklearn import linear_model, preprocessing
from sklearn.metrics import log_loss
import functools
import numpy as np
import jax.numpy as jnp

def process_latents(latents):
    if latents.dtype in [np.int64, np.int32, jnp.int64, jnp.int32]:
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)
        latents = one_hot_encoder.fit_transform(latents)
    elif latents.dtype in [np.float32, np.float64, jnp.float32, jnp.float64]:
        standardizer = preprocessing.StandardScaler()
        latents = standardizer.fit_transform(latents)
    else:
        raise ValueError(f'latents.dtype {latents.dtype} not supported')
    return latents


def linear_regression(X, y):
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.dtype in [np.float32, np.float64]
    assert y.dtype in [np.float32, np.float64]

    model = linear_model.LinearRegression(
        fit_intercept=True,
        n_jobs=-1,
        positive=False
    )

    model.fit(X, y)
    y_hat = model.predict(X)

    variance = 1 / 2

    average_negative_log_likelihood = -1 / 2 * np.log(2 * np.pi * variance) - 1 / (2 * variance) * np.mean(
        np.square(y - y_hat))
    return average_negative_log_likelihood, model.score(X, y)


@functools.lru_cache(maxsize=None)
def marginal_entropy_regression(labels, null_shape):
    #labels = torch.tensor(labels)
    null = np.zeros(null_shape)
    return linear_regression(null, labels)


def explicitness_regression(latents, sources):
    standardizer = preprocessing.StandardScaler()
    sources = standardizer.fit_transform(sources)
    latents = process_latents(latents)

    normalized_predictive_information_per_source = []

    for i_source in range(sources.shape[1]):
        source = sources[:, i_source]

        predictive_conditional_entropy, coefficient_of_determination = linear_regression(latents, source)

        normalized_predictive_information_per_source.append(coefficient_of_determination)

    return np.mean(np.array(normalized_predictive_information_per_source))


def logistic_regression(X, y):
    X = np.array(X)
    y = np.array(y)
    assert X.shape[0] == y.shape[0]
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.dtype in [np.float32, np.float64]
    assert y.dtype in [np.int32, np.int64]

    model = linear_model.LogisticRegression(
        penalty=None,
        dual=False,
        tol=1e-4,
        fit_intercept=True,
        class_weight='balanced',
        solver='lbfgs',
        max_iter=100,
        multi_class='multinomial',
        n_jobs=-1,
    )

    model.fit(X, y)
    logits = model.predict_log_proba(X)
    average_cross_entropy = np.mean(log_loss(y, logits))

    return average_cross_entropy


@functools.lru_cache(maxsize=None)
def marginal_entropy_classification(labels, null_shape):
    #labels = torch.tensor(labels)
    null = np.zeros(null_shape)
    return logistic_regression(null, labels)


def explicitness_classification(latents, sources):
    label_encoder = preprocessing.LabelEncoder()
    latents = process_latents(latents)

    normalized_predictive_information_per_source = []

    for i_source in range(sources.shape[1]):
        source = sources[:, i_source]
        labels = label_encoder.fit_transform(source)

        predictive_conditional_entropy = logistic_regression(latents, labels)
        marginal_source_entropy = marginal_entropy_classification(tuple(labels), latents.shape)

        normalized_predictive_information_per_source.append(
            (marginal_source_entropy - predictive_conditional_entropy) / marginal_source_entropy
        )

    return np.mean(np.array(normalized_predictive_information_per_source))