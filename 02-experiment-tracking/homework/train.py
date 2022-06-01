import argparse
import os
import pickle
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_valid)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)

if __name__ == '__main__':
    mlflow.set_tracking_uri("postgresql://mlflow:ur1CNZd2cvbCroaSwL0g@mlflow-backend-db.c4viu9aw88cp.ap-south-1.rds.amazonaws.com:5432/mlflow_db")
    mlflow.set_experiment("nyc-taxi-experiment")
    with mlflow.start_run():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data_path",
            default="./output",
            help="the location where the processed NYC taxi trip data was saved."
        )
        mlflow.autolog()
    
        args = parser.parse_args()
        
        run(args.data_path)
