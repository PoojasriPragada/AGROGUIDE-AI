from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the brain you just trained
# Ensure this matches the path in your Success message
model = pickle.load(open('models/crop_model.pkl', 'rb'))

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request, 
    N: float = Form(...), P: float = Form(...), K: float = Form(...),
    temp: float = Form(...), hum: float = Form(...), 
    ph: float = Form(...), rain: float = Form(...)
):
    # Convert input to the format the model expects
    input_data = np.array([[N, P, K, temp, hum, ph, rain]])
    prediction = model.predict(input_data)[0]
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction_text": f"The most suitable crop is: {prediction.upper()}"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)