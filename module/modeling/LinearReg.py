import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

def do_modeling(scaled_df, target, test_size=0.25, shuffle=False):
    # Split dataset (Independent / Target)
    X = scaled_df.drop(columns=[target]).values
    y = scaled_df[target].values

    # Split dataset (train / test)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=1-test_size,test_size=test_size,shuffle=shuffle)

    # Linear regression -> fit & predict
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred=reg.predict(X_test)
    
    # Evaluation
    print("<Result Score(training: ",str(1-test_size),", testing: ",str(test_size),")>", sep="")
    print("-->",np.round(reg.score(X_test, y_test),5))
    print()