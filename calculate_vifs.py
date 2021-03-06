import pandas as pd
print('pandas: {}'.format(pd.__version__))
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed

# Generate vifs and dataframe for inspection
def view_vif_(X):
    variables = [X.columns[i] for i in range(X.shape[1])]
    print(len(variables))
    vif = Parallel(n_jobs=-1,verbose=5)(delayed(variance_inflation_factor)(X[variables].values
                            , ix) for ix in range(len(variables)))
    
    # assemble df
    cols = X.columns.to_list()
    factors = pd.DataFrame(
        {'feature': cols,
         'vif_factor': vif
        })
    
    return factors


# Drop features with vif value greater than thresh number you can input
def drop_vif_(X, thresh=5.0):
    variables = [X.columns[i] for i in range(X.shape[1])]
    dropped=True
    while dropped:
        dropped=False
        print(len(variables))
        vif = Parallel(n_jobs=-1,verbose=5)(delayed(variance_inflation_factor)(X[variables].values, ix) for ix in range(len(variables)))

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(' dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables.pop(maxloc)
            dropped=True

    print('Remaining variables:')
    print([variables])
    return X[[i for i in variables]]


# EXAMPLE FUNCTION CALL
numerical_columns = df.select_dtypes('number').columns
categorical_columns = [c for c in df.columns if c not in numerical_columns]

mylist = list(numerical_columns)
mylist.remove("uniqueID")   #remove unique ID's if you have them

trainNumericNonNull = trainDataAndPredictions[mylist] # Selecting your data
trainNumericNonNull = trainNumericNonNull.dropna() #subset the dataframe

vif = view_vif_(trainNumericNonNull)
vif.sort_values('vif_factor', ascending=False)
