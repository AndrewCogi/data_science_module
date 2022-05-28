# data_science_module
It focused on easy and fast use by modularization from data preprocessing to modeling.

### module 설명
module에는 총 7가지 module이 존재합니다

### architecture
* #### model
    * data_preprocess

       > LabelEncoder
       > MinMaxScaler
       > RobustScaler
       > StandardScaler

    * modeling

        > DecisionTree
        > KNN
        > LinearReg

* 즉, from model.data_preprocess import LabelEncoder as le 처럼 사용이 가능합니다.

### 모듈 설명

* #### LabelEncoder
  * input parameter : org_df
   * org_df (type: DataFrame) >> LabelEncoding할 대상
    output : LabelEncoding을 마친 DataFrame 리턴

* #### MinMaxScaler

  * input parameter : org_df, target, showPlot
   * org_df (type: DataFrame) >> MinMaxScaling할 대상
   * target (type: String) >> target value의 feature명
   * showPlot (type: bool, default=False) >> 결과를 plotting할지에 대한 여부
    output : MinMaxScaling을 마친 DataFrame 리턴. plotting 여부에 따라 그래프를 그려줌

* #### RobustScaler

  * input parameter : org_df, target, showPlot
   * org_df (type: DataFrame) >> RobustScaling할 대상
   * target (type: String) >> target value의 feature명
   * showPlot (type: bool, default=False) >> 결과를 plotting할지에 대한 여부
    output : RobustScaling을 마친 DataFrame 리턴. plotting 여부에 따라 그래프를 그려줌

* #### StandardScaler

  * input parameter : org_df, target, showPlot
   * org_df (type: DataFrame) >> StandardScaling할 대상
   * target (type: String) >> target value의 feature명
   * showPlot (type: Bool, default=False) >> 결과를 plotting할지에 대한 여부
    output : StandardScaling을 마친 DataFrame 리턴. plotting 여부에 따라 그래프를 그려줌

* #### DecisionTree

  * input parameter : scaled_df, target, test_size, shuffle, criterion, showPlot
   * scaled_df (type: DataFrame) >> Scaling을 마친, DecisionTree대상
   * target (type: String) >> target value의 feature명
   * test_size (type: Float, default: 0.25) >> train_test_split할 때, testset 비율 지정
   * shuffle (type: Bool, default: False) >> train_test_split할 때, shuffle 여부 지정
   * criterion (type: String, default: 'gini') >> DecisionTree에서 사용할 criterion 종류 결정
   * showPlot (type: bool, default=False) >> 결과를 plotting할지에 대한 여부
    output : DecisionTree score를 보여주고 confision matrix도 나타내줌. plotting여부에 따라 tree를 그려줌

* #### KNN

  * input parameter : scaled_df, target, test_size, shuffle, k
   * scaled_df (type: DataFrame) >> Scaling을 마친, DecisionTree대상
   * target (type: String) >> target value의 feature명
   * test_size (type: Float, default: 0.25) >> train_test_split할 때, testset 비율 지정
   * shuffle (type: Bool, default: False) >> train_test_split할 때, shuffle 여부 지정
   * k (type: Int, default: 3) >> KNN에서 n_neighbors에 대한 값 지정
    output : KNN score를 cross_validation=5로 평가한 값을 보여주고 confision matrix도 나타내줌

* #### LinearReg

  * input parameter : scaled_df, target, test_size, shuffle
   * scaled_df (type: DataFrame) >> Scaling을 마친, DecisionTree대상
   * target (type: String) >> target value의 feature명
   * test_size (type: Float, default: 0.25) >> train_test_split할 때, testset 비율 지정
   * shuffle (type: Bool, default: False) >> train_test_split할 때, shuffle 여부 지정
    output : regression score를 보여줌
