# NUMERICAL FEATURES FUNCTIONS

import pandas as pd
print('pandas: {}'.format(pd.__version__))
# display all columns wide
pd.set_option('display.max_columns', None)
#turn off scientific notation
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import numpy as np
print('numpy: {}'.format(np.__version__))
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
print('seaborn: {}'.format(sns.__version__))
# Load and format colorblind palette
color_pal = sns.color_palette("colorblind", 6).as_hex()
colorblind6 = ','.join(color_pal).split(",")
# Use white grid plot background from seaborn
sns.set(font_scale=1.5, style="whitegrid")
from typing import List, Callable, Any
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# numeric feature generation
from numpy.random import seed
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
from sklearn import metrics
# for yeo-johnson normal transforms
from sklearn.preprocessing import PowerTransformer
# for RankGauss transforms
from sklearn.preprocessing import QuantileTransformer
# for D'Agostino K^2 normality test
from scipy.stats import normaltest
# for Anderson Darling normality test
from scipy.stats import anderson

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

# D’Agostino’s K^2 test is available via the normaltest() SciPy function
def dagostino(array, signif=0.05):
    seed(123)
    # normality test
    stat, p = normaltest(array)
#     print("D'Agostino's K^2 Statistics=%.3f, p=%.3f" % (stat, p))
    
    # interpret
    alpha = signif
    if p > alpha:
#         print("Sample looks Gaussian (fail to reject H0)")
        return True
    else:
#         print("Sample does not look Gaussian (reject H0)")
        return False
        

# Anderson-Darling Test is a modified version of a more sophisticated 
# nonparametric goodness-of-fit test called the Kolmogorov-Smirnov test.
# The anderson() SciPy function implements the Anderson-Darling test
def anderson_darling(array, signif=5, critical=2):
    # norm cvs: 25%, 10%, 5%, 2.5%, 1%, 0.5%
    # corresponding to 1-tail confidence: 75%, 90%, 95%, 97.5%, 99%, 99.5%
    # Default 95% significance level = 3rd in list critical-values
    seed(123)
    # normality test
    result = anderson(array, dist='norm')
#     print('Anderson-Darling Statistic: %.3f' % result.statistic)
    
    sl, cv = signif, result.critical_values[critical]
    if result.statistic < cv:
#         print(f"{100-sl}th: {cv}, data looks normal (fail to reject H0)")
        return True
    else:
#         print(f"{100-sl}th: {cv}, data does not look normal (reject H0)")
        return False
    
    
def best_transform(name, array_orig, array_log=None, array_yeo=None, array_rgauss=None
                   , skew_threshold = 0.5, kurt_threshold = 0.5, all_positive = False):

    # original skew, kurtosis
    skew_orig = array_orig.skew()
    kurt_orig = array_orig.kurtosis()
    
    #If skewness is larger than threshold and If yes, apply appropriate transformation
    if ((-1*skew_threshold > skew_orig) or (skew_threshold < skew_orig)
           or (-1*kurt_threshold > kurt_orig) or (kurt_threshold < kurt_orig)) :
    
        print(f"\n{name}: A skew of {np.round(skew_orig,2)}\
              means the distribution is not symmetric")
        print(f"{name}: A kurtosis of {np.round(kurt_orig,2)}\
              means the distribution tails are too narrow or too wide")
        # yeo-johnson transformed skew, kurtosis
        skew_yeo = array_yeo.skew()
        kurt_yeo = array_yeo.kurtosis()
        # rank-gauss transformed skew, kurtosis
        skew_rgauss = array_rgauss.skew()
        kurt_rgauss = array_rgauss.kurtosis()
        # log transformed skew, kurtosis
        if all_positive:
            skew_log = array_log.skew()
            kurt_log = array_log.kurtosis()
        else:
            skew_log = np.nan
            kurt_log = np.nan

        # original normality tests 
        dtest_orig = dagostino(array_orig)
        atest_orig = anderson_darling(array_orig)
        # yeo-johnson transformed normality tests 
        dtest_yeo = dagostino(array_yeo)
        atest_yeo = anderson_darling(array_yeo)
        # rank-gauss transformed normality tests 
        dtest_rgauss = dagostino(array_rgauss)
        atest_rgauss = anderson_darling(array_rgauss)
        # log transformed normality tests 
        if all_positive:
            dtest_log = dagostino(array_log)
            atest_log = anderson_darling(array_log)
        else:
            dtest_log = False
            atest_log = False
        

        # Automatic feature transform suggestions
        if (dtest_orig==True and atest_orig==True):
            print("CHOSE ORIG based on test results")
            return {'transform':'orig', 'skew_orig': skew_orig, 'kurt_orig': kurt_orig
                    , 'skew_log':skew_log, 'kurt_log':kurt_log
                    , 'skew_yeo':skew_yeo, 'kurt_yeo':kurt_yeo
                    , 'skew_rgauss': skew_rgauss, 'kurt_rgauss':kurt_rgauss}
        elif (all_positive and (dtest_log and atest_log)):
            print("CHOSE LOG based on test results")
            return {'transform':'log', 'skew_orig': skew_orig, 'kurt_orig': kurt_orig
                    , 'skew_log':skew_log, 'kurt_log':kurt_log
                    , 'skew_yeo':skew_yeo, 'kurt_yeo':kurt_yeo
                    , 'skew_rgauss': skew_rgauss, 'kurt_rgauss':kurt_rgauss}
        elif (dtest_rgauss and atest_rgauss):
            print("CHOSE RGAUSS based on test results")
            return {'transform':'rgauss', 'skew_orig': skew_orig, 'kurt_orig': kurt_orig
                    , 'skew_log':skew_log, 'kurt_log':kurt_log
                    , 'skew_yeo':skew_yeo, 'kurt_yeo':kurt_yeo
                    , 'skew_rgauss': skew_rgauss, 'kurt_rgauss':kurt_rgauss}
        elif (dtest_yeo and atest_yeo and skew_yeo > 0):
            print("CHOSE YEO based on test results")
            return {'transform':'yeo', 'skew_orig': skew_orig, 'kurt_orig': kurt_orig
                    , 'skew_log':skew_log, 'kurt_log':kurt_log
                    , 'skew_yeo':skew_yeo, 'kurt_yeo':kurt_yeo
                    , 'skew_rgauss': skew_rgauss, 'kurt_rgauss':kurt_rgauss}
        elif ((abs(skew_orig) + abs(kurt_orig) <= abs(skew_rgauss) + abs(kurt_rgauss) + 0.001) 
              and (skew_orig + kurt_orig <= skew_yeo + kurt_yeo + 0.001)):
            print("CHOSE ORIG based on skew and kurtosis")
            return {'transform':'orig', 'skew_orig': skew_orig, 'kurt_orig': kurt_orig
                    , 'skew_log':skew_log, 'kurt_log':kurt_log
                    , 'skew_yeo':skew_yeo, 'kurt_yeo':kurt_yeo
                    , 'skew_rgauss': skew_rgauss, 'kurt_rgauss':kurt_rgauss}
        elif ((all_positive and 
               (abs(skew_log) + abs(kurt_log) <= abs(skew_yeo) + abs(kurt_yeo) + 0.01))):
            print("CHOSE LOG based on skew and kurtosis")
            return {'transform':'log', 'skew_orig': skew_orig, 'kurt_orig': kurt_orig
                    , 'skew_log':skew_log, 'kurt_log':kurt_log
                    , 'skew_yeo':skew_yeo, 'kurt_yeo':kurt_yeo
                    , 'skew_rgauss': skew_rgauss, 'kurt_rgauss':kurt_rgauss}
        elif (skew_yeo > 0 and 
              (abs(skew_yeo) + abs(kurt_yeo) <= abs(skew_rgauss) + abs(kurt_rgauss) + 0.01)):
            print("CHOSE YEO based on skew and kurtosis")
            return {'transform':'yeo', 'skew_orig': skew_orig, 'kurt_orig': kurt_orig
                    , 'skew_log':skew_log, 'kurt_log':kurt_log
                    , 'skew_yeo':skew_yeo, 'kurt_yeo':kurt_yeo
                    , 'skew_rgauss': skew_rgauss, 'kurt_rgauss':kurt_rgauss}
        else:
            print("CHOSE RGAUSS based on skew and kurtosis")
            return {'transform':'rgauss', 'skew_orig': skew_orig, 'kurt_orig': kurt_orig
                    , 'skew_log':skew_log, 'kurt_log':kurt_log
                    , 'skew_yeo':skew_yeo, 'kurt_yeo':kurt_yeo
                    , 'skew_rgauss': skew_rgauss, 'kurt_rgauss':kurt_rgauss}
    
    else:
        print("CHOSE ORIG based on no conditions")
        return {'transform':'orig', 'skew_orig': skew_orig, 'kurt_orig': kurt_orig
                , 'skew_log':None, 'kurt_log':None
                , 'skew_yeo':None, 'kurt_yeo':None
                , 'skew_rgauss':None, 'kurt_rgauss':None}


def numeric_feature_autotransform(DF, plot = False
                       , skew_threshold = 0.5
                       , kurt_threshold = 0.5):
    """
    Function to auto-transform a numerical pandas dataframe into approx. 
    normally-distributed numeric feature columns based on thresholds.
    Note: pandasDF.kurtosis is same as scipy.stats.stats.kurtosis(series, bias=False)
          , so normal kurtosis=3 for usual range [2,4] but pandas kurtosis=0.
    Note: If either skew or kurtosis outside [-0.5, 0.5], then perform transforms
    
    Input: 
        DF = pandas dataframe with numerical feature columns
        skew_threshold, kurt_threshold = values to detect non-normal distributions
        plot = True if you want to view the plots
        
    Output:  
        matplotlib plots per column of before and after transformation distributions
        print-out which transformation chosen and why
    
    Returns:
        pandas dataframe with new transformed columns col_yeoj, col_rgauss if transform needed;
            otherwise if no transform needed the return dataframe will be empty
        
    """
    # make local copy of local_df
    local_df = DF.copy()
    
    #instatiate PowerTransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=True,) 
    #instantiate RankGauss Transformer
    rankgt = QuantileTransformer(n_quantiles=100, random_state=123, output_distribution='normal')
    
    # Helper function fit_transform yeo_johnson
    # Works on positive, negative, and 0's
    def apply_yeo_johnson(sublocal_df):
        #Fit the data to the powertransformer
        skl_yeojohnson = pt.fit(sublocal_df)
        #Get the Lambdas that were found
        calc_lambdas = skl_yeojohnson.lambdas_
        #Transform the data 
        skl_yeojohnson = pt.transform(sublocal_df)
        return skl_yeojohnson
    
    
    # Go through columns in DataFrame
    col_names = local_df.columns.values
    
    for col in col_names:
        #Get column skewness
        skew = local_df[col].skew()
        kurtosis = local_df[col].kurtosis()
        transformed = True
        
        # check pandas series values are all positive?
        all_positive = (pd.Series(local_df.loc[:, col]) >=0).all()
        if not all_positive:
            print(f"col: {col} has negative values")
        
        if plot:
            #Plot original distribution
            sns.set_style("darkgrid")
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
            ax1 = sns.distplot(local_df[col].dropna()
                               , kde=1, norm_hist=1
                               , ax=axes[0], color=colorblind6[0])
            ax1.set(xlabel='Original ' + col)
        
        #If skewness is larger than threshold and If yes, apply appropriate transformation
        if ((-1*skew_threshold > skew) or (skew_threshold < skew)
               or (-1*kurt_threshold > kurtosis) or (kurt_threshold < kurtosis)) :

            # Apply yeo-johnson transformation
            skl_yeojohnson = apply_yeo_johnson(local_df.loc[:, [col]])
            local_df[f"{col.lower()}_yeoj"] = skl_yeojohnson
        
            # Apply rankGauss transformation
            rankgt.fit(local_df.loc[:, [col]])
            rankg = rankgt.transform(local_df.loc[:, [col]])
            local_df[f"{col.lower()}_rgauss"] = rankg
            
            # calc log(x +1) to avoid undefined log(0), only works if all values >=0
            if all_positive:
                local_df[f"{col.lower()}_log"] = np.log(local_df.loc[:, [col]] +1)
            else:
                local_df[f"{col.lower()}_log"] = np.nan
        
        else:
            print(f"\n NO TRANSFORMATION APPLIED FOR {col}: A skew of {np.round(skew, 2)} and kurtosis of {np.round(kurtosis,2)} means the distribution is approx. normal")
            transformed = False

            
        # Auto-suggestion which transform best 
        if transformed: 
            transform_sel_dict = best_transform(col, local_df.loc[:, col]
                           , local_df[f"{col.lower()}_log"]
                           , local_df[f"{col.lower()}_yeoj"]
                           , local_df[f"{col.lower()}_rgauss"]
                           , skew_threshold, kurt_threshold
                           , all_positive)
            print(f"selected transform: {transform_sel_dict['transform']}")
            print(f"\nYeo-J Transformation yielded skewness of {transform_sel_dict['skew_yeo']}")
            print(f"Yeo-J Transformation yielded kurtosis of {transform_sel_dict['kurt_yeo']}")
            print(f"\nRank-Gauss Transformation yielded skewness of {transform_sel_dict['skew_rgauss']}")
            print(f"Rank-Gauss Transformation yielded kurtosis of {transform_sel_dict['kurt_rgauss']}")
            if all_positive:
                print(f"\nLog Transformation yielded skewness of {transform_sel_dict['skew_log']}")
                print(f"Log Transformation yielded kurtosis of {transform_sel_dict['kurt_log']}")
        
        
        # drop the original columns
        # return just transformed columns; returnd df empty if no transformations happened
        #local_df.drop(col, axis=1, inplace=True)
        
        #Compare before and after if plot is True
        if plot:
            print('\n ------------------------------------------------------')     
            if transformed:           
                sns.set_palette("Paired")
                ax2 = sns.distplot(local_df[f"{col.lower()}_log"].dropna()
                                   , kde=1, norm_hist=1
                                   , ax=axes[1], color=colorblind6[1])
                ax3 = sns.distplot(local_df[f"{col.lower()}_yeoj"].dropna()
                                   , kde=1, norm_hist=1
                                   , ax=axes[2], color=colorblind6[2])
                ax4 = sns.distplot(local_df[f"{col.lower()}_rgauss"].dropna()
                                   , kde=1, norm_hist=1
                                   , ax=axes[3], color=colorblind6[3])
                ax2.set(xlabel='Log Transformed ' + col)
                ax3.set(xlabel='Yeo-Johnson Transformed ' + col)
                ax4.set(xlabel='RankGauss Transformed ' + col)
                plt.show()
            else:
                # No transformation applied
                ax2.set(xlabel='NO TRANSFORM ' + col)
                ax3.set(xlabel='NO TRANSFORM ' + col)
                ax4.set(xlabel='NO TRANSFORM ' + col)
                plt.show()
    
    return local_df
 
 
# EXAMPLE FUNCTION CALL QUICK NUMERICAL FEATURES
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

                      
# EXAMPLE FUNCTION CALL AUTO-NUMERICAL FEATURES
g = features.copy()
mylist_numerical.remove("uniqueID")

# After aggregations, do transforms
# np.warnings.filterwarnings('ignore')
dfs_result_agg = numeric_feature_autotransform(g[mylist_numerical]
                                   , plot = True
                                   , skew_threshold = 0.5
                                   , kurt_threshold = 0.5)

# append just new numerical features as right-side columns of features
logger.info(f"shape before new numerical features is {features.shape}")
newfeatures = [x for x in dfs_result_agg.columns if x not in features.columns]
features = pd.concat([features, dfs_result_agg[newfeatures]], axis=1)
logger.info(f"shape after new numerical features is {features.shape}")

# double-check you did the right thing
print(features.head(2))
cols = features.columns.tolist()
cols
