import pandas as pd

from typing import Optional
from fastapi import FastAPI
from DataModel import DataModel
from DataModel_Train import DataModel_train

from PredictionModel1 import Model1
from PredictionModel2 import Model2
from PredictionModel3 import Model3

from fastapi.responses import JSONResponse

from joblib import dump

app = FastAPI()

@app.post("/predictModel1")
def make_prediction(dataModel : DataModel):
   df2 = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
   model = Model1()
   result = model.model_make_predictions(df2)
   return result[0]

@app.post("/predictModel2")
def make_prediction(dataModel : DataModel):
   df2 = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
   model = Model2()
   result = model.model_make_predictions(df2)
   return result[0]

@app.post("/predictModel3")
def make_prediction(dataModel : DataModel):
   df2 = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
   model = Model3()
   result = model.model_make_predictions(df2)
   return result[0]

@app.post("/predictionsModel1")
def make_predictions(dataModel : list[DataModel]):
   model = Model1()
   all_data = []
   i = 0
   for data in dataModel:
      df2 = pd.DataFrame(data.dict(), columns=data.dict().keys(), index=[0])
      result = model.model_make_predictions(df2)
      all_data.append({ 'respuesta {0}'.format(i) : result.tolist()[0]})
      i+=1
   return JSONResponse(all_data)

@app.post("/predictionsModel2")
def make_predictions(dataModel : list[DataModel]):
   model = Model2()
   all_data = []
   i = 0
   for data in dataModel:
      df2 = pd.DataFrame(data.dict(), columns=data.dict().keys(), index=[0])
      result = model.model_make_predictions(df2)
      all_data.append({ 'respuesta {0}'.format(i) : result.tolist()[0]})
      i+=1
   return JSONResponse(all_data)

@app.post("/predictionsModel3")
def make_predictions(dataModel : list[DataModel]):
   model = Model3()
   all_data = []
   i = 0
   for data in dataModel:
      df2 = pd.DataFrame(data.dict(), columns=data.dict().keys(), index=[0])
      result = model.model_make_predictions(df2)
      all_data.append({ 'respuesta {0}'.format(i) : result.tolist()[0]})
      i+=1
   return JSONResponse(all_data)

@app.post("/trainModel1")
def train_M1 (dataModel : list[DataModel_train]):
   model = Model1()
   df2 = pd.DataFrame([x.dict() for x in dataModel])
   X = df2.drop('Admission_Points', axis = 1)
   y = df2['Admission_Points']
   model.train(X,y)
   filename = 'pipelines/artifacts/polynomial/PipelineM1.joblib'
   dump(model.model, filename)
   rta = model.score()
   return rta

@app.post("/trainModel2")
def train_M2 (dataModel : list[DataModel_train]):
   model = Model2()
   df2 = pd.DataFrame([x.dict() for x in dataModel])
   X = df2.drop('Admission_Points', axis = 1)
   y = df2['Admission_Points']
   model.train(X,y)
   filename = 'pipelines/artifacts/polynomial/PipelineM2.joblib'
   dump(model.model, filename)
   rta = model.score()
   return rta

@app.post("/trainModel3")
def train_M3 (dataModel : list[DataModel_train]):
   model = Model3()
   df2 = pd.DataFrame([x.dict() for x in dataModel])
   X = df2.drop('Admission_Points', axis = 1)
   y = df2['Admission_Points']
   model.train(X,y)
   filename = 'pipelines/artifacts/polynomial/PipelineM3.joblib'
   dump(model.model, filename)
   rta = model.score()
   return rta

@app.get("/scoreModel1")
def get_scoreM1 ():
   model = Model1()
   rta = model.score()
   return rta

@app.get("/scoreModel2")
def get_scoreM2 ():
   model = Model2()
   rta = model.score()
   return rta

@app.get("/scoreModel3")
def get_scoreM3 ():
   model = Model3()
   rta = model.score()
   return rta
