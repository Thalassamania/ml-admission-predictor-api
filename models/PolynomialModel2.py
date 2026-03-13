from joblib import load
import pandas as pd
from sklearn.metrics import mean_absolute_error

class PolyDegree10Model:

    def __init__(self):
        self.model = load("pipelines/artifacts/polynomial/PipelinePolyDegree10.joblib")
    
    def model_make_predictions(self,data):
        result = self.model.predict(data)
        return result

    def train(self, X, y):
        result = self.model.fit(X,y)
        return result
    
    def score(self):
        raw_train = pd.read_csv('pipelines/data/train.csv')
        raw_train = raw_train.drop_duplicates()
        raw_train = raw_train.dropna()
        X = raw_train.drop('Admission Points', axis = 1)
        y = raw_train['Admission Points']
        rta = self.model.score(X,y)
        y_predicted = self.model.predict(X)
        rta2 = mean_absolute_error(y, y_predicted)
        result = [rta, rta2]
        return result
