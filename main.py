from fastapi import FastAPI
from typing import List
from pydantic import BaseModel, Field, validator
from ml.model import load_model
import pandas as pd
import numpy as np


# Encoding categorical data using the fitted encoders
encoder = load_model("./model/encoder.joblib")
lb = load_model("./model/label_binarizer.joblib")
mdoel = load_model("./model/ML_model.pkl")
data = pd.read_csv("./Data/census_cleaned.csv")
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
# Declare the data object with its components and their type.
class Inputdata(BaseModel):
    age: List[int]
    workclass: List[str]
    fnlgt: List[int]
    education: List[str]
    education_num: List[int] = Field(..., alias="education-num")  # Alias to map the column name, ... makes this field required
    marital_status: List[str] = Field(..., alias="marital-status") 
    occupation: List[str]
    relationship: List[str]
    race: List[str]
    sex: List[str]
    capital_gain: List[int] = Field(..., alias="capital-gain")
    capital_loss: List[int] = Field(..., alias="capital-loss")
    hours_per_week: List[int] = Field(..., alias="hours-per-week")
    native_country: List[str] = Field(..., alias="native-country")

    @validator('age')
    def age_range(cls,v):
        if any(i<=16 for i in v):
            raise ValueError('Age should be above 16')
        else:
            return v
    @validator('workclass')
    def check_workclass(cls, v):
        if not all(i in data['workclass'].unique() for i in v):
            raise ValueError(f"workclass must be one of the following: {', '.join(data['workclass'].unique())}")
        return v
    @validator('fnlgt')
    def check_fnlgt(cls, v):
        if any(i<=1 for i in v):
            raise ValueError('fnlgt should be above 1')
        else:
            return v
    @validator('education')
    def check_education(cls, v):
        if not all(i in data['education'].unique() for i in v):
            raise ValueError(f"education must be one of the following: {', '.join(data['education'].unique())}")
        return v
    @validator('education_num')
    def check_education_num(cls, v):
        if any((i<1)or(i>16) for i in v):
            raise ValueError('education_num should be between 1 and 16')
        return v
    @validator('marital_status')
    def check_marital_status(cls, v):
        if not all(i in data['marital-status'].unique() for i in v):
            raise ValueError(f"marital-status must be one of the following: {', '.join(data['marital-status'].unique())}")
        return v
    @validator('occupation')
    def check_occupation(cls, v):
        if not all(i in data['occupation'].unique() for i in v):
            raise ValueError(f"occupation must be one of the following: {', '.join(data['occupation'].unique())}")
        return v
    @validator('relationship')
    def check_relationship(cls, v):
        if not all(i in data['relationship'].unique() for i in v):
            raise ValueError(f"relationship must be one of the following: {', '.join(data['relationship'].unique())}")
        return v
    @validator('race')
    def check_race(cls, v):
        if not all(i in data['race'].unique() for i in v):
            raise ValueError(f"race must be one of the following: {', '.join(data['race'].unique())}")
        return v
    @validator('sex')
    def check_sex(cls, v):
        if not all(i in data['sex'].unique() for i in v):
            raise ValueError(f"sex must be one of the following: {', '.join(data['sex'].unique())}")
        return v
    @validator('capital_gain')
    def check_capital_gain(cls, v):
        if any((i<0)or(i>99999) for i in v):
            raise ValueError('capital_gain should be between 0 and 99999')
        return v
    @validator('capital_loss')
    def check_capital_loss(cls, v):
        if any(i<0 for i in v):
            raise ValueError('capital_loss should not be lower than 0')
        return v
    @validator('hours_per_week')
    def check_hours_per_week(cls, v):
        if any((i<0)or(i>168) for i in v):
            raise ValueError('hours_per_week should be between 0 and 168')
        return v
    @validator('native_country')
    def check_native_country(cls, v):
        if not all(i in data['native-country'].unique() for i in v):
            raise ValueError(f"native_country must be one of the following: {', '.join(data['native-country'].unique())}")
        return v
    
# Initialize FastAPI instance
app = FastAPI()


# GET on the root giving a welcome message.
@app.get("/")
async def welcome():
    return {"Root": "Welcome to the webpage!"}


# POST that does model inference
@app.post("/inference/")
async def model_inference(item: Inputdata):
    X = pd.DataFrame(item.dict()).rename(columns={'marital_status': 'marital-status', 'native_country': 'native-country'})
    X_categorical = X[cat_features].values
    X_continuous = X.drop(*[cat_features], axis=1)
    X_categorical = encoder.transform(X_categorical)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    output = model.predict(X)
    return {'Result':output.tolist()}


