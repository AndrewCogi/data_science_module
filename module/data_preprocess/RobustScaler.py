import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

''' RobustScaler
input parameter : org_df, target, showPlot
org_df (type: DataFrame) >> Target of RobustScaling
target (type: String) >> Feature name of target value
showPlot (type: bool, default=False) >> Whether to plot the result
output : Return DataFrame that after RobustScaling. Drawing graph based on whether or not plotting
'''
def do_scaling(org_df, target, showPlot=False):
    # Init variables
    scaler = RobustScaler()
    title = 'Robust Scaling'

    # Scaling
    X_df=org_df.drop([target], axis=1)
    scaled_df = scaler.fit_transform(X_df)
    scaled_df = pd.DataFrame(scaled_df,columns=list(X_df.columns))

    # show plot
    if(showPlot == True):
        import seaborn as sns
        from matplotlib import pyplot as plt
        # Make subplot
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,8))

        # Before Scaling plot
        ax1.set_title("Before Scaling")
        ax1.set_xlabel("values")
        for feature in list(X_df.columns):
            sns.kdeplot(X_df[feature], ax=ax1)

        # After Scaling plot
        ax2.set_title(title)
        ax2.set_xlabel("values")
        for feature in list(scaled_df.columns):
            sns.kdeplot(scaled_df[feature], ax=ax2)
        plt.show()

    # Make complete scaled dataframe
    scaled_df = pd.concat([scaled_df,org_df[target]], axis=1)
    return scaled_df