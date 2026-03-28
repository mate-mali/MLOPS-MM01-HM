from pydantic import BaseModel

class ze_input(BaseModel):
    text: str

class ze_output(BaseModel):
    prediction: str

    