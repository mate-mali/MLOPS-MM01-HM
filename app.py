from fastapi import FastAPI
import uvicorn
from inference import load_classifier, load_sentence_transformer, map_prediction_to_sentiment
from api.models.sentiment import ze_input, ze_output

#initiate fastapi app
app = FastAPI()

#generic endpoint to welcome people
@app.get("/")
def welcome_root():
    return {"message": "For better curling experience, please use /docs endpoint "}


#actual predicting endpoint, it was tested in english and is quite hard to get neutral sentimen out of it
@app.post("/predict")
def predict_sentiment(input: ze_input): 
    '''Endpoint to predict sentiment of the input text\
    Args:        input (ze_input): Input text for sentiment analysis that is provided on post request'''
    classifier = load_classifier()
    sentence_transformer = load_sentence_transformer()
    text_embedding = sentence_transformer.encode(input.text)
    prediction = classifier.predict([text_embedding])
    #added som minor intermiary dictionary fro lookup fro values 0-2
    ze_sentiment = map_prediction_to_sentiment(prediction[0])
    return ze_output(prediction=ze_sentiment)