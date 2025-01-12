# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import logging
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, slice_performance
import joblib
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
logger.info("Reading the data")
data = pd.read_csv("./Data/census_cleaned.csv")
# Optional enhancement, use K-fold cross validation instead
# of a train-test split.
logger.info("Processing the data")
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Save encoders
joblib.dump(encoder, "model/encoder.joblib")
joblib.dump(lb, "model/label_binarizer.joblib")

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
logger.info("Training the model")
model = train_model(X_train, y_train)
# Do inference
logger.info("Doing inference")
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"The model performance is with precision {precision}, recall {recall} and F1 score {fbeta}")
# Slice performance
logger.info("Slice performance for categorical varaible 'workclass'")
output = slice_performance(['workclass'], test, y_test, preds)
print(output)

# Save the trained model using joblib
logger.info("Save the model")
joblib.dump(model, 'model/ML_model.pkl')
print("Model saved as 'ML_model.pkl'")
