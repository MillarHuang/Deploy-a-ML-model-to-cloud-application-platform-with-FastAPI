import pandas as pd
import numpy as np
import pytest
import joblib
from sklearn.ensemble import RandomForestClassifier
from ..model import train_model, inference, compute_model_metrics


@pytest.fixture
def data():
    """ Sample data to test"""
    df = pd.DataFrame({
        'age': [54, 28],
        'workclass': ['Private', 'Private'],
        'fnlgt': [308087, 167062],
        'education': ['Some-college', 'Some-college'],
        'education-num': [10, 10],
        'marital-status': ['Married-civ-spouse', 'Divorced'],
        'occupation': ['Tech-support', 'Adm-clerical'],
        'relationship': ['Husband', 'Unmarried'],
        'race': ['White', 'White'],
        'sex': ['Male', 'Male'],
        'capital-gain': [0, 0],
        'capital-loss': [1977, 0],
        'hours-per-week': [18, 40],
        'native-country': ['United-States', 'United-States'],
        'salary': ['>50K', '<=50K']}
    )
    # Encoding categorical data using the fitted encoders
    encoder = joblib.load("./model/encoder.joblib")
    lb = joblib.load("./model/label_binarizer.joblib")
    y = df['salary']
    X = df.drop(['salary'], axis=1)
    categorical_features = [
                            "workclass",
                            "education",
                            "marital-status",
                            "occupation",
                            "relationship",
                            "race",
                            "sex",
                            "native-country"]
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    y = lb.transform(y.values).ravel()
    return X, y


@pytest.fixture
def model():
    """ The well-trained model"""
    model = joblib.load('./model/ML_model.pkl')
    return model


def test_model_type(data):
    """ Test if the train_model function will output the expected Random Forest model from sklearn"""
    X, y = data
    model_output = train_model(X, y)
    assert isinstance(model_output, RandomForestClassifier), "Output model is not a Random Forest model trained from sklearn."


def test_inference(model, data):
    """ Test to see if the inference function output is as the model inference output"""
    X, y = data
    assert (inference(model, X) == model.predict(X)).all(), "The inference output is not consistent with the model prediction result."
    assert isinstance(inference(model, X), np.ndarray), "The inference output is not np.ndarray"


def test_compute_model_metrics(model, data):
    """ Test to see if the compute_model_metrics function output is as expected(length,type)"""
    X, y = data
    preds = model.predict(X)
    assert all(isinstance(element, float) for element in compute_model_metrics(y, preds)), "The output of compute_model_metrics is not all of float type."
    assert len(compute_model_metrics(y, preds)) == 3, "Expect to have three outputs from compute_model_metrics function."
