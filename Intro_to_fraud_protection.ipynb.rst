
Setup and Data Import

.. code:: ipython3

    Thanks to Kevin Jacobs

.. code:: ipython3

    import pandas as pd
    import numpy as np
    

.. code:: ipython3

    # Read the csv file
    df = pd.read_csv('creditcard.csv')

.. code:: ipython3

    print(df)


.. parsed-literal::

                Time         V1         V2        V3        V4        V5  \
    0            0.0  -1.359807  -0.072781  2.536347  1.378155 -0.338321   
    1            0.0   1.191857   0.266151  0.166480  0.448154  0.060018   
    2            1.0  -1.358354  -1.340163  1.773209  0.379780 -0.503198   
    3            1.0  -0.966272  -0.185226  1.792993 -0.863291 -0.010309   
    4            2.0  -1.158233   0.877737  1.548718  0.403034 -0.407193   
    ...          ...        ...        ...       ...       ...       ...   
    284802  172786.0 -11.881118  10.071785 -9.834783 -2.066656 -5.364473   
    284803  172787.0  -0.732789  -0.055080  2.035030 -0.738589  0.868229   
    284804  172788.0   1.919565  -0.301254 -3.249640 -0.557828  2.630515   
    284805  172788.0  -0.240440   0.530483  0.702510  0.689799 -0.377961   
    284806  172792.0  -0.533413  -0.189733  0.703337 -0.506271 -0.012546   
    
                  V6        V7        V8        V9  ...       V21       V22  \
    0       0.462388  0.239599  0.098698  0.363787  ... -0.018307  0.277838   
    1      -0.082361 -0.078803  0.085102 -0.255425  ... -0.225775 -0.638672   
    2       1.800499  0.791461  0.247676 -1.514654  ...  0.247998  0.771679   
    3       1.247203  0.237609  0.377436 -1.387024  ... -0.108300  0.005274   
    4       0.095921  0.592941 -0.270533  0.817739  ... -0.009431  0.798278   
    ...          ...       ...       ...       ...  ...       ...       ...   
    284802 -2.606837 -4.918215  7.305334  1.914428  ...  0.213454  0.111864   
    284803  1.058415  0.024330  0.294869  0.584800  ...  0.214205  0.924384   
    284804  3.031260 -0.296827  0.708417  0.432454  ...  0.232045  0.578229   
    284805  0.623708 -0.686180  0.679145  0.392087  ...  0.265245  0.800049   
    284806 -0.649617  1.577006 -0.414650  0.486180  ...  0.261057  0.643078   
    
                 V23       V24       V25       V26       V27       V28  Amount  \
    0      -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053  149.62   
    1       0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724    2.69   
    2       0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752  378.66   
    3      -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458  123.50   
    4      -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   69.99   
    ...          ...       ...       ...       ...       ...       ...     ...   
    284802  1.014480 -0.509348  1.436807  0.250034  0.943651  0.823731    0.77   
    284803  0.012463 -1.016226 -0.606624 -0.395255  0.068472 -0.053527   24.79   
    284804 -0.037501  0.640134  0.265745 -0.087371  0.004455 -0.026561   67.88   
    284805 -0.163298  0.123205 -0.569159  0.546668  0.108821  0.104533   10.00   
    284806  0.376777  0.008797 -0.473649 -0.818267 -0.002415  0.013649  217.00   
    
            Class  
    0           0  
    1           0  
    2           0  
    3           0  
    4           0  
    ...       ...  
    284802      0  
    284803      0  
    284804      0  
    284805      0  
    284806      0  
    
    [284807 rows x 31 columns]
    

.. code:: ipython3

    print(df.describe())


.. parsed-literal::

                    Time            V1            V2            V3            V4  \
    count  284807.000000  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean    94813.859575  3.919560e-15  5.688174e-16 -8.769071e-15  2.782312e-15   
    std     47488.145955  1.958696e+00  1.651309e+00  1.516255e+00  1.415869e+00   
    min         0.000000 -5.640751e+01 -7.271573e+01 -4.832559e+01 -5.683171e+00   
    25%     54201.500000 -9.203734e-01 -5.985499e-01 -8.903648e-01 -8.486401e-01   
    50%     84692.000000  1.810880e-02  6.548556e-02  1.798463e-01 -1.984653e-02   
    75%    139320.500000  1.315642e+00  8.037239e-01  1.027196e+00  7.433413e-01   
    max    172792.000000  2.454930e+00  2.205773e+01  9.382558e+00  1.687534e+01   
    
                     V5            V6            V7            V8            V9  \
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean  -1.552563e-15  2.010663e-15 -1.694249e-15 -1.927028e-16 -3.137024e-15   
    std    1.380247e+00  1.332271e+00  1.237094e+00  1.194353e+00  1.098632e+00   
    min   -1.137433e+02 -2.616051e+01 -4.355724e+01 -7.321672e+01 -1.343407e+01   
    25%   -6.915971e-01 -7.682956e-01 -5.540759e-01 -2.086297e-01 -6.430976e-01   
    50%   -5.433583e-02 -2.741871e-01  4.010308e-02  2.235804e-02 -5.142873e-02   
    75%    6.119264e-01  3.985649e-01  5.704361e-01  3.273459e-01  5.971390e-01   
    max    3.480167e+01  7.330163e+01  1.205895e+02  2.000721e+01  1.559499e+01   
    
           ...           V21           V22           V23           V24  \
    count  ...  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean   ...  1.537294e-16  7.959909e-16  5.367590e-16  4.458112e-15   
    std    ...  7.345240e-01  7.257016e-01  6.244603e-01  6.056471e-01   
    min    ... -3.483038e+01 -1.093314e+01 -4.480774e+01 -2.836627e+00   
    25%    ... -2.283949e-01 -5.423504e-01 -1.618463e-01 -3.545861e-01   
    50%    ... -2.945017e-02  6.781943e-03 -1.119293e-02  4.097606e-02   
    75%    ...  1.863772e-01  5.285536e-01  1.476421e-01  4.395266e-01   
    max    ...  2.720284e+01  1.050309e+01  2.252841e+01  4.584549e+00   
    
                    V25           V26           V27           V28         Amount  \
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  284807.000000   
    mean   1.453003e-15  1.699104e-15 -3.660161e-16 -1.206049e-16      88.349619   
    std    5.212781e-01  4.822270e-01  4.036325e-01  3.300833e-01     250.120109   
    min   -1.029540e+01 -2.604551e+00 -2.256568e+01 -1.543008e+01       0.000000   
    25%   -3.171451e-01 -3.269839e-01 -7.083953e-02 -5.295979e-02       5.600000   
    50%    1.659350e-02 -5.213911e-02  1.342146e-03  1.124383e-02      22.000000   
    75%    3.507156e-01  2.409522e-01  9.104512e-02  7.827995e-02      77.165000   
    max    7.519589e+00  3.517346e+00  3.161220e+01  3.384781e+01   25691.160000   
    
                   Class  
    count  284807.000000  
    mean        0.001727  
    std         0.041527  
    min         0.000000  
    25%         0.000000  
    50%         0.000000  
    75%         0.000000  
    max         1.000000  
    
    [8 rows x 31 columns]
    

.. code:: ipython3

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    import pandas as pd

.. code:: ipython3

    data = pd.read_csv('creditcard.csv')

.. code:: ipython3

    # Only use the 'Amount' and 'V1', ..., 'V28' features
    features = ['Amount'] + ['V%d' % number for number in range(1, 29)]

.. code:: ipython3

    # The target variable which we would like to predict, is the 'Class' variable
    target = 'Class'

.. code:: ipython3

    # Now create an X variable (containing the features) and an y variable (containing only the target variable)
    X = data[features]
    y = data[target]

.. code:: ipython3

    def normalize(X):
        """
        Make the distribution of the values of each variable similar by subtracting the mean and by dividing by the standard deviation.
        """
        for feature in X.columns:
            X[feature] -= X[feature].mean()
            X[feature] /= X[feature].std()
        return X

.. code:: ipython3

    # Define the model
    model = LogisticRegression()
    
    # Define the splitter for splitting the data in a train set and a test set
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    
    # Loop through the splits (only one)
    for train_indices, test_indices in splitter.split(X, y):
        # Select the train and test data
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
        
        # Normalize the data
        X_train = normalize(X_train)
        X_test = normalize(X_test)
        
        # Fit and predict!
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # And finally: show the results
        print(classification_report(y_test, y_pred))


.. parsed-literal::

    C:\Users\Peter\Miniconda3\envs\fraud\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      
    C:\Users\Peter\Miniconda3\envs\fraud\lib\site-packages\ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys
    C:\Users\Peter\Miniconda3\envs\fraud\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    

.. parsed-literal::

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00    142158
               1       0.88      0.61      0.72       246
    
        accuracy                           1.00    142404
       macro avg       0.94      0.81      0.86    142404
    weighted avg       1.00      1.00      1.00    142404
    
    

