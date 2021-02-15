'''module dockstring'''
import sys
import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

def get_data(path):
    '''module dockstring'''
    df = pd.read_csv(path)
    y = df.pop("cons_general").to_numpy()
    y[y< 4] = 0
    y[y>= 4] = 1
    X = df.to_numpy()
    return X, y, df

def preproc(X):
    '''module dockstring'''
    X = preprocessing.scale(X)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X = imp.transform(X)
    return X

def train_predict(X, y, penalty, C, solver):
    '''module dockstring'''
    # Linear model
    clf = LogisticRegression(penalty=penalty, C=C, solver=solver)
    yhat = cross_val_predict(clf, X, y, cv=10)
    return yhat, clf

def mlflow_log(df, y, yhat, penalty, C, solver, clf):
    '''module dockstring'''
    with mlflow.start_run():

        acc = np.mean(yhat==y)
        tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        mlflow.log_param('penalty', penalty)
        mlflow.log_param('C', C)
        mlflow.log_param('solver', solver)
        mlflow.log_metric('acc', acc)
        mlflow.log_metric('specificity', specificity)
        mlflow.log_metric('sensitivity', sensitivity)
        mlflow.sklearn.log_model(clf, "model")

        # Now print to file
        with open("metrics.json", 'w') as outfile:
            json.dump({
                "accuracy": acc,
                "specificity": specificity,
                "sensitivity":sensitivity
                }, outfile)
        # Let's visualize within several slices of the dataset
        score = yhat == y
        score_int = [int(s) for s in score]
        df['pred_accuracy'] = score_int

        # Bar plot by region
        sns.set_color_codes("dark")
        ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette = "Greens_d")
        ax.set(xlabel="Region", ylabel = "Model accuracy")
        plt.savefig("by_region.png", dpi=80)
        mlflow.log_artifact("./by_region.png")

if __name__ == "__main__":

    X, y, df = get_data("data_processed.csv")

    X = preproc(X)

    REMOTE_SERVER_URI = 'http://ml_flow:5000'
    mlflow.set_tracking_uri(REMOTE_SERVER_URI)
    PENALTY = str(sys.argv[1]) if len(sys.argv) > 1 else 'l2'
    C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    SOLVER = str(sys.argv[3]) if len(sys.argv) > 3 else 'lbfgs'

    y_hat, clf = train_predict(X=X, y=y, penalty=PENALTY, C=C, solver=SOLVER)

    mlflow_log(df=df, y=y, yhat=y_hat, penalty=PENALTY, C=C, solver=SOLVER, clf=clf)
