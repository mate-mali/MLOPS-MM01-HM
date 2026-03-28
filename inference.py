import joblib
import sentence_transformers as st


#wrapper fro laoding model and returning it using joblin
def load_classifier():
    #load classifier using joblib
    model = joblib.load("model/classifier.joblib")
    return model

#wrapper fro loading mdoel transformer from folder
def load_sentence_transformer():
    #load fodler with this transformer thingy
    model = st.SentenceTransformer('model/sentence_transformer.model')
    return model

#thrwo results of classifier into dictionary and return text
def map_prediction_to_sentiment(prediction):
    preds = {0: "negative", 1: "neutral", 2: "positive"}
    return preds.get(prediction, "Not possible to classify")