# data_science_module
It focused on easy and fast use by modularization from data preprocessing to modeling.

### Module Information

* #### LabelEncoder
  * input parameter : org_df
   * org_df (type: DataFrame) >> Target of LabelEncoding
  * output : Return DataFrame after LabelEncoding

* #### MinMaxScaler

  * input parameter : org_df, target, showPlot
   * org_df (type: DataFrame) >> Target of MinMaxScaling
   * target (type: String) >> Feature name of target value
   * showPlot (type: bool, default=False) >> Whether to plot the result
  * output : Return DataFrame that after MinMaxScaling. Drawing graph based on whether or not plotting

* #### RobustScaler

  * input parameter : org_df, target, showPlot
   * org_df (type: DataFrame) >> Target of RobustScaling
   * target (type: String) >> Feature name of target value
   * showPlot (type: bool, default=False) >> Whether to plot the result
  * output : Return DataFrame that after RobustScaling. Drawing graph based on whether or not plotting

* #### StandardScaler

  * input parameter : org_df, target, showPlot
   * org_df (type: DataFrame) >> Target of StandardScaling
   * target (type: String) >> Feature name of target value
   * showPlot (type: Bool, default=False) >> Whether to plot the result
  * output : Return DataFrame that after StandardScaling. Drawing graph based on whether or not plotting

* #### DecisionTree

  * input parameter : scaled_df, target, test_size, shuffle, criterion, showPlot
   * scaled_df (type: DataFrame) >> Target of DecisionTree that after Scaling
   * target (type: String) >> Feature name of target value
   * test_size (type: Float, default: 0.25) >> Specify testset ratio when training_test_split
   * shuffle (type: Bool, default: False) >> Specify whether shuffle when training_test_split
   * criterion (type: String, default: 'gini') >> Determine the type of criterion used in Decision Tree
   * showPlot (type: bool, default=False) >> Whether to plot the result
  * output : Show DecisionTree score and confision matrix. Drawing tree based on whether or not plotting

* #### KNN

  * input parameter : scaled_df, target, test_size, shuffle, k
   * scaled_df (type: DataFrame) >> Target of DecisionTree that after Scaling
   * target (type: String) >> Feature name of target value
   * test_size (type: Float, default: 0.25) >> Specify testset ratio when training_test_split
   * shuffle (type: Bool, default: False) >> Specify whether shuffle when training_test_split
   * k (type: Int, default: 3) >> Specify a value for n_neighbors in KNN
  * output : Show KNN score evaluated by cross_validation=5 and confision matrix

* #### LinearReg

  * input parameter : scaled_df, target, test_size, shuffle
   * scaled_df (type: DataFrame) >> Target of DecisionTree that after Scaling
   * target (type: String) >> Feature name of target value
   * test_size (type: Float, default: 0.25) >> Specify testset ratio when training_test_split
   * shuffle (type: Bool, default: False) >> Specify whether shuffle when training_test_split
  * output : Show regression score
