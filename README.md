# Admission Points Prediction API

## Overview
This project exposes a set of machine-learning models for predicting graduate admission scores through a REST API. It builds on the linear regression work of a previous laboratory and demonstrates how to prepare data, train models, and serve them via a FastAPI application.

The data and feature engineering are the same as in the earlier lab, with only minor variable name changes. The key innovation here is the construction of a scikit-learn pipeline and its integration into a deployable service. The pipeline performs two preprocessing steps:

- Imputer: replaces missing values with the mean of the corresponding column.
- Standard scaler: scales numerical features so that models that assume normally distributed inputs (e.g., linear or polynomial regression) perform better.

The notebooks in the `pipelines/notebooks/` folder explore additional steps such as `ordinal_encoder` and `OneHotEncoder` for categorical variables. These are demonstrated for completeness but are not required because all features in the admissions dataset are numeric.

## Models and Selection
Several regression algorithms were evaluated, including decision trees, random forests, robust regression, Gaussian process regression, support-vector regression, and polynomial regression. After experimentation, the team selected polynomial regression of degree 12 because it achieved the best performance on the training data. Three variants of this model are packaged as separate pipelines (Model 1, Model 2, and Model 3). Each pipeline is persisted to disk with `joblib` and loaded by the API on demand.

Performance was evaluated using multiple metrics. The API returns two metrics alongside each score request: explained variance score and mean absolute error (MAE). Explained variance values close to 1 indicate that the predicted and true values are almost indistinguishable, while MAE gives an average error in the scale of the original data. During model selection, the polynomial regression pipeline achieved scores above 0.9 on the explained variance metric.

## Running the API
1. Clone this repository.
2. Open the project in your preferred IDE.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

You may also install them manually with `pip install fastapi uvicorn scikit-learn pandas joblib`.

4. Start the API locally:

```bash
uvicorn main:app --reload
```

This command launches the FastAPI application using Uvicorn. By default it serves on `http://127.0.0.1:8000`.

5. Open the interactive documentation at `http://127.0.0.1:8000/docs`. FastAPI automatically generates Swagger UI where you can explore and test each endpoint.

If any of the models return poor predictions (e.g., because the persistent pipelines are missing), first execute the Jupyter notebooks under `pipelines/notebooks/` to train the models and generate the `.joblib` artifacts.

## API Endpoints
The API exposes endpoints for each of the three models. Replace `{i}` with 1, 2, or 3 depending on which model you want to use.

| Endpoint               | Method | Description                                                |
| ---------------------- | ------ | ---------------------------------------------------------- |
| `/predictModel{i}`     | POST   | Predicts a single admission score. Expects a JSON body     |
|                        |        | with the input features defined in `DataModel`. Returns a  |
|                        |        | single float representing the predicted admission points.  |
| `/predictionsModel{i}` | POST   | Predicts admission scores for multiple inputs. Expects a   |
|                        |        | JSON list of `DataModel` objects and returns a list of      |
|                        |        | predictions.                                               |
| `/trainModel{i}`       | POST   | Retrains the pipeline using new labeled data. The request  |
|                        |        | body must be a list of `DataModel_train` objects (which     |
|                        |        | include the target `Admission_Points`). The endpoint       |
|                        |        | stores the updated model to the `Pipeline/` folder and     |
|                        |        | returns updated performance metrics.                       |
| `/scoreModel{i}`       | GET    | Returns evaluation metrics for the selected model without  |
|                        |        | retraining. The response contains the explained variance   |
|                        |        | score and mean absolute error.                             |

The model index `{i}` allows you to compare different pipeline configurations. For example, `/predictModel1` and `/scoreModel1` operate on the first pipeline, while `/trainModel2` retrains the second pipeline.

## Data Models
**DataModel**  
Defines the input schema for prediction endpoints. It contains numerical features such as GRE score, TOEFL score, university rating, SOP, LOR, CGPA, and research indicator. These fields must be provided in the same order as the training data.

**DataModel_train**  
Extends `DataModel` by including an additional `Admission_Points` field (the target). This model is used when retraining a pipeline via `/trainModel{i}`.

## Pipeline and Experiments
The Jupyter notebooks in the `pipelines/notebooks/` folder document the preprocessing and model selection process. The notebooks explore several regression algorithms and hyperparameters, ultimately selecting a 12-degree polynomial regression model because it consistently produced the highest explained variance scores. The notebooks also record the values of various evaluation metrics so you can reproduce the selection process.

## Conclusion
This project demonstrates how to take a machine-learning pipeline from exploration to production. By wrapping the trained models in a FastAPI service and using Pydantic for validation, the API provides a simple interface for making predictions, retraining models, and inspecting model performance. The final polynomial regression model achieved an explained variance score close to 0.99 on the training data, indicating that the features used are highly predictive of admission outcomes. While there is always room for further experimentation and additional data, this pipeline and API serve as a solid foundation for decision support in admissions processes.
