from fastapi import FastAPI
import joblib
import uvicorn
import sentence_transformers as st
from api.models.sentiment import ze_input, ze_output


app = FastAPI()

def load_classifier():
    model = joblib.load("model/classifier.joblib")
    return model

def load_sentence_transformer():
    model = st.SentenceTransformer('model/sentence_transformer.model')
    return model

def map_prediction_to_sentiment(prediction):
    preds = {0: "negative", 1: "neutral", 2: "positive"}
    return preds.get(prediction, "unknown")


@app.get("/")
def welcome_root():
    return {"message": "Mateusz Malinowski fancy szmancy API"}

@app.post("/predict")
def predict_sentiment(input: ze_input): 
    classifier = load_classifier()
    sentence_transformer = load_sentence_transformer()
    text_embedding = sentence_transformer.encode(input.text)
    prediction = classifier.predict([text_embedding])
    ze_sentiment = map_prediction_to_sentiment(prediction[0])
    return ze_output(prediction=ze_sentiment)