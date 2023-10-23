import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# เลือกวิธีการวัดผล
from sklearn.metrics import accuracy_score,f1_score
# เรียก Library แบ่งข้อมูล
from sklearn.model_selection import train_test_split
# Normalize
from sklearn.preprocessing import Normalizer,StandardScaler
import mlflow
from mlflow import log_metric, log_param, log_params, log_artifacts
from mlflow.models import infer_signature
from urllib.parse import urlparse
from mlflow.data.pandas_dataset import PandasDataset

# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

def get_data(sheeturl):
    dat = pd.read_csv(sheeturl)
    sdat = dat.iloc[:,1:7] #เลือกทุก Row และเอาเฉพาะแถวที่ 1 ถึง 6
    sdat.columns = ['height','weight','age','sleep','somtum','gender']
    data = sdat
    return data

def process_data(data):
    # Drop ข้อมูลที่เว้นว่างทั้งแถวออก
    data = data.dropna()
    
    #กำหนด Input Feature
    X = data.iloc[:,:5]
    # กำหนด Output feature
    y = data['gender']
    return X,y

def model_train(X,y):
    max_depth = 2
    dataset: PandasDataset = mlflow.data.from_pandas(data, source=sheeturl)
    with mlflow.start_run():
        mlflow.log_param("max depth",max_depth)
        mlflow.log_input(dataset, context="training")
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3,random_state=0)
        
        model = RandomForestClassifier(random_state=0, max_depth=max_depth)
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        acc = accuracy_score(y_test,pred)
        f1 = f1_score(y_test,pred)
        log_metric("accuracy", acc)
        log_metric("f1 score", f1)
        print(f"The accuracy of model is {acc} and f1 score is {f1}")
        # signature = infer_signature(X_test, pred)
        mlflow.sklearn.log_model(model,"model",registered_model_name="RandomForestModel")

if __name__=='__main__':
    sheeturl = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQKwisRA0C_9rBJQbgalt63C9aIFe3DluapSnq0ULVtRnYiOP0uSXVaYbXOPwhupIZm7gwVCOqqDCnw/pub?gid=1101564382&single=true&output=csv'
    data = get_data(sheeturl)
    X,y = process_data(data)
    model_train(X,y)



##############
# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

# import logging
# import sys
# import warnings
# from urllib.parse import urlparse

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import ElasticNet
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split

# import mlflow
# import mlflow.sklearn
# from mlflow.models import infer_signature

# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)


# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2


# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     np.random.seed(40)

#     # Read the wine-quality csv file from the URL
#     csv_url = (
#         "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
#     )
#     try:
#         data = pd.read_csv(csv_url, sep=";")
#     except Exception as e:
#         logger.exception(
#             "Unable to download training & test CSV, check your internet connection. Error: %s", e
#         )

#     # Split the data into training and test sets. (0.75, 0.25) split.
#     train, test = train_test_split(data)

#     # The predicted column is "quality" which is a scalar from [3, 9]
#     train_x = train.drop(["quality"], axis=1)
#     test_x = test.drop(["quality"], axis=1)
#     train_y = train[["quality"]]
#     test_y = test[["quality"]]

#     alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
#     l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

#     with mlflow.start_run():
#         lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
#         lr.fit(train_x, train_y)

#         predicted_qualities = lr.predict(test_x)

#         (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

#         print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
#         print(f"  RMSE: {rmse}")
#         print(f"  MAE: {mae}")
#         print(f"  R2: {r2}")

#         mlflow.log_param("alpha", alpha)
#         mlflow.log_param("l1_ratio", l1_ratio)
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_metric("r2", r2)
#         mlflow.log_metric("mae", mae)

#         predictions = lr.predict(train_x)
#         signature = infer_signature(train_x, predictions)

#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

#         # Model registry does not work with file store
#         if tracking_url_type_store != "file":
#             # Register the model
#             # There are other ways to use the Model Registry, which depends on the use case,
#             # please refer to the doc for more information:
#             # https://mlflow.org/docs/latest/model-registry.html#api-workflow
#             mlflow.sklearn.log_model(
#                 lr, "model", registered_model_name="ElasticnetWineModel", signature=signature
#             )
#         else:
#             mlflow.sklearn.log_model(lr, "model", signature=signature)
