import requests
import json

# POST request on the live API
data = {
        'age': [54],
        'workclass': ['Private'],
        'fnlgt': [308087],
        'education': ['Some-college'],
        'education-num': [10],
        'marital-status': ['Married-civ-spouse'],
        'occupation': ['Adm-clerical'],
        'relationship': ['Husband'],
        'race': ['White'],
        'sex': ['Male'],
        'capital-gain': [0],
        'capital-loss': [0],
        'hours-per-week': [40],
        'native-country': ['United-States']}
r = requests.post("https://salary-classification-qvyc.onrender.com/inference/", data=json.dumps(data))
print(r.json())

# Get request on the live API
r = requests.get("https://salary-classification-qvyc.onrender.com/")
print(r.json())
