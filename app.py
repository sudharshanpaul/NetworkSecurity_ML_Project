import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv
mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

client = pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # <-- Fix: "allow_header" should be "allow_headers"
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/",tags=['authentication'])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is Successful")
    except Exception as e:
        logging.error(f"Training Pipeline Failed: {e}", exc_info=True)
        return Response(f"Training failed: {str(e)}", status_code=500)
    
@app.post("/predict")
async def predict_route(request:Request,file:UploadFile=File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_models/preprocessor.pkl")
        final_model = load_object("final_models/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor,model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predictes_column'] = y_pred
        print(df['predictes_column'])

        df.to_csv("predicted_output/output.csv")

        table_html = df.to_html(classes='table table-stripped')

        return templates.TemplateResponse("table.html",{"request":request,"table":table_html})
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
if __name__=='__main__':
    app_run(app,host="localhost",port=8000)