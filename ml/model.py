from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Initialize the SVC model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using
    precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.svm._classes.SVC
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """
    The function to save the model to certain path

    Inputs
    ------
    model : 
        The model to save
    path : str
        The path to save
    """
    joblib.dump(model, path)


def load_model(path):
    """
    The function to load the model

    Inputs
    ------
    path : str
        The path where model saved
    """
    model = joblib.load(path)
    return model 


def slice_performance(slices, data_test, y_test, preds):
    """
    The function that outputs the performances (precision,
    recall, and F1) of the model on slices of the data

    Inputs
    ------
    slices : list
        The list of (categorical) variables for for slicing
    data_test : pd.DataFrame
        The data used to test the performance
    y_test : np.array
        The label of the test data
    preds : np.array
        The prediction on the test data
    Returns
    -------
    output : pd.DataFrame
        The output performance of the model on slices of the data
    """
    df = data_test.copy()
    df['label'] = y_test
    df['preds'] = preds
    output = (
        df.groupby(slices)[['label', 'preds']]
        .apply(lambda x: pd.Series(compute_model_metrics(x['label'],
                                                         x['preds']),
                                   index=['f1_score',
                                          'precision',
                                          'recall']))
        .reset_index()
    )
    return output
