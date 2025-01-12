# Deploy-a-ML-model-to-cloud-application-platform-with-FastAPI


## Introduction
In this repo, it includes three parts:

* Build a CI/CD pipeline: 
    - CI: only when the code committed passes the tests, it can create a pull request to integrate to the main branch.
    - CD: when the code is integrated to the main branch, it will automatically deploy the API on Render.

* Build a Machine learning model to classify the salary condition based on the census data provided, the code is provided in train_model.py.

* Create an API based on the well-trained ML model on Render. The example code to do GET/POST on the live API is provided in API_request_sample.py.


## Structure
#### Model and Data:
1. Data/
    * Clean_data.ipynb: the jupyternotebook file that is used to clean the raw data census.csv, and generate census_cleaned.csv.
    * census.csv: raw dataset.
    * census_cleaned.csv: cleaned dataset.

2. ml/
    * data.py: contains the function to preprocess the data.
    * model.py: contains the functions to train a random forest model, do inference, get performance metrics and slice performance.
    * tests/test_model.py: the unit tests to test the functions in model.py.

3. model/
    * ML_model.pkl: the random forest model trained on census_cleaned.csv dataset.
    * encoder.joblib: categorical encoder fitted on training set.
    * label_binarizer.joblib: binary encoder fitted on training set.

4. train_model.py: 
    - Script to preprocess the data, train a random forest model and save it at model/, and calculate the performance of the well-trained model.

5. model_card_template.md:
    - the model card for the well-trained random forest model.

#### API and Deployment
1. main.py:
    - create an API using FastAPI, with POST method to get the inference result of the ML model.

2. test_API.py:
    - the unit test to test the GET and POST method of API.

3. API_request_sample.py:
    - the example code to use the requests module to do GET/POST on the live API.

4. sanitycheck.py:
    - this script will scan the test cases written for the GET() and POST() APIs in test_API.py, and generate a report that lists any problems it detects with the test cases.

#### Others
1. requirements.txt: the dependencies of this API.

2. .github/workflows/CI.yml:
    - the CI and CD pipeline.
   
   