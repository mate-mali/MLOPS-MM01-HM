from pydantic import BaseModel


class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    def to_numpy(self):
        return [[
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width
        ]]  #i am adding this because this damned base model is not working with input dict into
        #the predict function, so i am giving up

class PredictResponse(BaseModel):
    prediction: str