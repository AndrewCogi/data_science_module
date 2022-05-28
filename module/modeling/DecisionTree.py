import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def do_modeling(scaled_df, target, test_size=0.25, shuffle=False, criterion='gini', showPlot=False):
    # Split dataset (Independent / Target)
    X = scaled_df.drop(columns=[target]).values
    y = scaled_df[target].values

    # Split dataset (train / test)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=1-test_size,test_size=test_size,shuffle=shuffle)
    
    # Decision tree -> fit & predict
    tr = DecisionTreeClassifier(criterion=criterion)
    tr.fit(X_train,y_train)
    y_pred_tr = tr.predict(X_test)
    
    if showPlot == True:
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        # Plotting
        plt.figure()
        plot_tree(tr,filled=True)
        plt.title("Decision Tree (training: "+str(1-test_size)+", testing: "+str(test_size)+")")
        plt.show()
    
    # Show matrix
    print("<Classification Report (Decisiion Tree)>")
    print(classification_report(tr.predict(X_test),y_test))

    # Evaluation
    print("<Result of test model(training: ",str(1-test_size),", testing: ",str(test_size),")>", sep="")
    print("<Result Score(training: ",str(1-test_size),", testing: ",str(test_size),")>", sep="")
    print('-->%.5f' % accuracy_score(y_test, y_pred_tr))
    print()