# Admission Points Prediction API

## Overview
This repository exposes a FastAPI service for predicting graduate admission scores with persisted scikit-learn pipelines. The project includes:

- a REST API in [main.py]
- Pydantic request schemas in [DataModel.py] and [DataModel_Train.py]
- trained model artifacts under `pipelines/artifacts/`
- experimentation notebooks under `pipelines/notebooks/`

The current API serves three models:

- polynomial regression, degree 2
- polynomial regression, degree 3
- support vector regression

## Project Structure

```text
.
|-- main.py
|-- DataModel.py
|-- DataModel_Train.py
|-- models/
|   |-- Polynomial/
|   |   |-- PolynomialModel2.py
|   |   `-- PolynomialModel3.py
|   `-- SVM/
|       `-- SVRModel.py
|-- pipelines/
|   |-- artifacts/
|   |-- data/
|   `-- notebooks/
`-- requirements.txt
```

## Installation
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the API
Start the development server with:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000` and the interactive docs at `http://127.0.0.1:8000/docs`.

## Request Schemas

### Prediction payload
The prediction endpoints accept the fields defined in `DataModel`. Example:

```json
{
  "serial_no": 1,
  "gre_score": 320,
  "toefl_score": 110,
  "university_rating": 4,
  "sop": 4.5,
  "lor": 4.0,
  "cgpa": 9.1,
  "research": 1
}
```

### Training payload
The training endpoints accept the same fields plus `Admission_Points`. Example:

```json
{
  "serial_no": 1,
  "gre_score": 320,
  "toefl_score": 110,
  "university_rating": 4,
  "sop": 4.5,
  "lor": 4.0,
  "cgpa": 9.1,
  "research": 1,
  "Admission_Points": 82.5
}
```

## API Endpoints

### Single prediction

- `POST /polynomial/degree2`
- `POST /polynomial/degree3`
- `POST /svm/svr`

Each endpoint receives one `DataModel` object and returns a single numeric prediction.

### Batch prediction

- `POST /polynomial/degree2/predictions`
- `POST /polynomial/degree3/predictions`
- `POST /svm/svr/predictions`

Each endpoint receives a list of `DataModel` objects and returns a list of predictions.

### Retraining

- `POST /polynomial/degree2/train`
- `POST /polynomial/degree3/train`
- `POST /svm/svr/train`

Each endpoint receives a list of `DataModel_train` objects, retrains the corresponding model, and writes the updated artifact to:

- `pipelines/artifacts/polynomial/PipelinePolyDegree2.joblib`
- `pipelines/artifacts/polynomial/PipelinePolyDegree3.joblib`
- `pipelines/artifacts/SVM/PipelineSVR.joblib`

### Scoring

- `GET /polynomial/degree2/score`
- `GET /polynomial/degree3/score`
- `GET /svm/svr/score`

Each score endpoint returns:

- model score from scikit-learn's `Pipeline.score(...)`
- mean absolute error on the dataset loaded by the model wrapper

## Training Data and Notebooks
The repository includes data files in `pipelines/data/` and exploratory notebooks in:

- `pipelines/notebooks/polynomial/`
- `pipelines/notebooks/SVM/`
- `pipelines/notebooks/linear/`
- `pipelines/notebooks/DataReview.ipynb`

These notebooks are used to study preprocessing choices, compare regressors, and export `.joblib` artifacts used by the API.
