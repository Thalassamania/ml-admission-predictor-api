import pandas as pd

from fastapi import FastAPI
from DataModel import DataModel
from DataModel_Train import DataModel_train

from models.Polynomial.PolynomialModel2 import PolyDegree2
from models.Polynomial.PolynomialModel3 import PolyDegree3
from models.SVM.SVRModel import SVRModel

from fastapi.responses import JSONResponse

from joblib import dump

app = FastAPI()

@app.post("/polynomial/degree2")
def make_prediction(dataModel : DataModel):
   df2 = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
   model = PolyDegree2()
   result = model.model_make_predictions(df2)
   return result[0]

@app.post("/polynomial/degree3")
def make_prediction(dataModel : DataModel):
   df2 = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
   model = PolyDegree3()
   result = model.model_make_predictions(df2)
   return result[0]

@app.post("/svm/svr")
def make_prediction(dataModel : DataModel):
   df2 = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
   model = SVRModel()
   result = model.model_make_predictions(df2)
   return result[0]

@app.post("/polynomial/degree2/predictions")
def make_predictions(dataModel : list[DataModel]):
   model = PolyDegree2()
   all_data = []
   i = 0
   for data in dataModel:
      df2 = pd.DataFrame(data.dict(), columns=data.dict().keys(), index=[0])
      result = model.model_make_predictions(df2)
      all_data.append({ 'respuesta {0}'.format(i) : result.tolist()[0]})
      i+=1
   return JSONResponse(all_data)

@app.post("/polynomial/degree3/predictions")
def make_predictions(dataModel : list[DataModel]):
   model = PolyDegree3()
   all_data = []
   i = 0
   for data in dataModel:
      df2 = pd.DataFrame(data.dict(), columns=data.dict().keys(), index=[0])
      result = model.model_make_predictions(df2)
      all_data.append({ 'respuesta {0}'.format(i) : result.tolist()[0]})
      i+=1
   return JSONResponse(all_data)

@app.post("/svm/svr/predictions")
def make_predictions(dataModel : list[DataModel]):
   model = SVRModel()
   all_data = []
   i = 0
   for data in dataModel:
      df2 = pd.DataFrame(data.dict(), columns=data.dict().keys(), index=[0])
      result = model.model_make_predictions(df2)
      all_data.append({ 'respuesta {0}'.format(i) : result.tolist()[0]})
      i+=1
   return JSONResponse(all_data)

@app.post("/polynomial/degree2/train")
def train (dataModel : list[DataModel_train]):
   model = PolyDegree2()
   df2 = pd.DataFrame([x.dict() for x in dataModel])
   X = df2.drop('Admission_Points', axis = 1)
   y = df2['Admission_Points']
   model.train(X,y)
   filename = 'pipelines/artifacts/polynomial/PipelinePolyDegree2.joblib'
   dump(model.model, filename)
   rta = model.score()
   return rta

@app.post("/polynomial/degree3/train")
def train (dataModel : list[DataModel_train]):
   model = PolyDegree3()
   df2 = pd.DataFrame([x.dict() for x in dataModel])
   X = df2.drop('Admission_Points', axis = 1)
   y = df2['Admission_Points']
   model.train(X,y)
   filename = 'pipelines/artifacts/polynomial/PipelinePolyDegree3.joblib'
   dump(model.model, filename)
   rta = model.score()
   return rta

@app.post("/svm/svr/train")
def train (dataModel : list[DataModel_train]):
   model = SVRModel()
   df2 = pd.DataFrame([x.dict() for x in dataModel])
   X = df2.drop('Admission_Points', axis = 1)
   y = df2['Admission_Points']
   model.train(X,y)
   filename = 'pipelines/artifacts/SVM/PipelineSVR.joblib'
   dump(model.model, filename)
   rta = model.score()
   return rta

@app.get("/polynomial/degree2/score")
def get_score():
   model = PolyDegree2()
   rta = model.score()
   return rta

@app.get("/polynomial/degree3/score")
def get_score():
   model = PolyDegree3()
   rta = model.score()
   return rta

@app.get("/svm/svr/score")
def get_score():
   model = SVRModel()
   rta = model.score()
   return rta