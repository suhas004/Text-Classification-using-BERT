from fastapi import FastAPI
import ktrain
import uvicorn


app = FastAPI()

predictor = ktrain.load_predictor('weights/')


@app.get("/")
def read_root():
    return {"Hello": "Suhas" }


@app.get("/hello_name/{name}")
def read_root(name: str):
    string_to_return = str("Hello {}".format(name))

    return {"string_to_return": string_to_return}



@app.get("/predict/{text}")
def read_item(text: str):
    prediction = predictor.predict(text)

    return {"prediction": prediction}



if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)

