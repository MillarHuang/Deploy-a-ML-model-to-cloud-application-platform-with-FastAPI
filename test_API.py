import json
from fastapi.testclient import TestClient
from main import app
"""
Run: python web_exercise1_test.py
"""
client = TestClient(app)

def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Root": "Welcome to the webpage!"}


def test_post_status():
    data = json.dumps({
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
        'native-country': ['United-States']})
    r = client.post("/inference/", data=data)
    print(r.json())
    assert r.status_code == 200


def test_post_predictions():
    data = json.dumps({
        'age': [54, 28],
        'workclass': ['Private', 'Private'],
        'fnlgt': [308087,167062],
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
        'native-country': ['United-States', 'United-States']})
    r = client.post("/inference/", data=data)
    print(r.json())
    assert r.json()['Result'][0] == 1
    assert r.json()['Result'][1] == 0
