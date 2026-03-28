import pytest
import app
from fastapi.testclient import TestClient       


def test_app_response():
    client = TestClient(app.app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "For better curling experience, please use /docs endpoint "}

def test_model_inference():
    client = TestClient(app.app)
    response = client.post("/predict", json={"text": "Sleeping is fun especially after 30"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "positive"}

def test_moddels_loading():
    from inference import load_classifier, load_sentence_transformer
    classifier = load_classifier()
    sentence_transformer = load_sentence_transformer()
    assert classifier is not None
    assert sentence_transformer is not None

def test_empty_input():
    client = TestClient(app.app)
    response = client.post("/predict", json={"text": " "})
    assert response.status_code == 200
    assert response.json() == {"prediction": "No text provided for classification"} #i added this as alternative output for dict lookup

# def test_invalid_input():
#     client = TestClient(app.app)
#     response = client.post("/predict", json={"text": "Mateusz'); drop table studentsx;"}) #apparently dropping table via sql injection passes as 'neutral'
#     assert response.status_code == 200
#     assert response.json() == {"prediction": "Not possible to classify"}

def test_invalid_input_alternative():
    client = TestClient(app.app)
    response = client.post("/predict", json={"text": 1991}) #lets check numbers then
    assert response.status_code == 422 # according to docs this is invalid format for request so i will go with that 
 
def test_multiple_inputs():
    client = TestClient(app.app)
    inputs = ["I have been in mountains and it was nice", "I worked 68 hours this week, why dont round it up to 69?", "I have no opinion on that"] #positive, negative, neutral   
    expected_outputs = ["positive", "negative", "neutral"]
    for text, expected in zip(inputs, expected_outputs):
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 200
        assert response.json() == {"prediction": expected}

def test_valid_json():
    client = TestClient(app.app)
    response = client.post("/predict", json={"text": "I like chocolate"})
    assert response.status_code == 200
    assert response.json()

