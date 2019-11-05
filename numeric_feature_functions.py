# NUMERICAL FEATURES FUNCTIONS

import pandas as pd
print('pandas: {}'.format(pd.__version__))
from typing import List, Callable, Any
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Apply parellel function
def apply_parallel(
            df_group: pd.DataFrame,
            mylist: List[str],
            function: Callable[[pd.DataFrame, List[str]], pd.DataFrame],
            n_jobs: int = 1,
            progress_bar: bool = False,
            **kwargs: Any
        ):
    """Function for running group by applys for Pandas in parallel

    Arguments:
        df_group {pd.DataFrame} -- Dataframe containing a group
            retrieved from a call to groupby
        function {Callable[[pd.DataFrame, List[str]], pd.DataFrame]}
            -- Function to be applied per group.

    Keyword Arguments:
        n_jobs {int} -- [description] (default: {1})
        progress_bar {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    if progress_bar:
        df_group = tqdm(df_group)


    result_list = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(function)(group, mylist, **kwargs) for name, group in df_group
    )
    return pd.concat(result_list)

# Generic function to calculate numeric features
def features_agg_numeric(
        g: pd.DataFrame,
        numeric_columns_list,
        ) -> pd.DataFrame:
    
    # Numeric covariates
    for numeric_col in numeric_columns_list:
        g[f"{numeric_col.lower()}_mean"]= g[numeric_col].mean()
        g[f"{numeric_col.lower()}_min"]= g[numeric_col].min()
        g[f"{numeric_col.lower()}_max"]= g[numeric_col].max()
        g[f"{numeric_col.lower()}_median"]= np.median(g[numeric_col])
        # logs
        g[f"{numeric_col.lower()}_logmean"]= np.log(g[numeric_col].mean())
        g[f"{numeric_col.lower()}_logmax"]= np.log(g[numeric_col].max())
        g[f"{numeric_col.lower()}_logmedian"]= np.log(g[numeric_col].median())
        
    return g
 
 
# EXAMPLE FUNCTION CALL
numerical_columns = df.select_dtypes('number').columns
categorical_columns = [c for c in df.columns if c not in numerical_columns]
mylist_numerical = list(numerical_columns)

g = features.copy()
g = g[mylist_numerical]
mylist_numerical.remove("uniqueID")

# aggregation if any
dfs_result_agg = apply_parallel(
    df_group = g.groupby('uniqueID'), # aggregation-level
    mylist = mylist_numerical,        # columns to be transformed
    function = features_agg_numeric,
    n_jobs = -1,                      # use all cores
    progress_bar = True
)

# append just new numerical features as right-side columns of features
newfeatures = [x for x in dfs_result_agg.columns if x not in features.columns]
features = pd.concat([features, dfs_result_agg[newfeatures]], axis=1)
logger.info(f"shape of features is {features.shape}")

# double-check you did the right thing
print(features.head(2))
cols = features.columns.tolist()
cols
