import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

''' LabelEncoder
input parameter : org_df
org_df (type: DataFrame) >> Target of LabelEncoding
output : Return DataFrame after LabelEncoding '''

def do_encoding(org_df):
    le = LabelEncoder()
    for feature in list(org_df.columns):
        org_df[feature]=le.fit_transform(org_df[feature].values)
    return org_df