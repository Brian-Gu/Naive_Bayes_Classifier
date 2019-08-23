
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Data Preparation

df_dev = pd.read_csv('propublicaTrain.csv')
df_tst = pd.read_csv('propublicaTest.csv')

# Univariate Analysis

varList = df_dev.dtypes
var_num = varList.index[varList.values == 'int64']

def univariate_num_var(df, vars):
    """
    :param df:   Input DataFrame
    :param vars: Numerical Variable List
    :return:     Summary Statistic Results
    """
    res = pd.DataFrame()
    for var in vars:
        sum_stat = df[var].describe().transpose()
        sum_stat["Variable"] = var
        sum_stat["Miss#"] = len(df) - sum_stat["count"]
        sum_stat["Miss%"] = sum_stat["Miss#"] * 100 / len(df)
        res = res.append(sum_stat, ignore_index=True)
    order = ['Variable', 'count', 'Miss#', 'Miss%', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    return res[order]

summary_num = univariate_num_var(df_dev, var_num)

# 2. Naive Bayes Classifier

def is_continuous(df, var):
    """
    :param df:  Train Set
    :param var: Variable of Interest
    :return:    Is a Continuous Var (True/False)
    """
    if len(np.unique(df[var]))/df[var].count() < 0.001:
        return False
    return True


def naive_bayes_classifier(train_x, train_y):
    """
    :param train_x: Train X Set
    :param train_y: Train Y Set
    :return:        Naive Bayes Classifier Parameters
    """
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    y = train_y.name
    df, df[y] = train_x.copy(), train_y
    params, params['prob'], params['bins'], params['nums'] = {}, {}, {}, []
    params['prior'] = df.groupby(y).size()/len(df)

    # Create Bins for Continuous Variable
    for v in train_x.columns:
        if is_continuous(df, v):
            params['nums'].append(v)
            df.loc[df[v] == df.loc[:, v].max(), v] = np.inf
            ser, bins = pd.qcut(df[v], 10, retbins=True, duplicates='drop')
            params['bins'][v] = list(np.insert(bins, 0, -np.inf))
            df[v] = pd.cut(df[v], bins=params['bins'][v])

    # Calculate Conditional Probability
    for v in train_x.columns:
        params['prob'][v] = df.groupby([y, v]).size()/df.groupby(y).size()
    return params


def naive_bayes_predictor(params, test_x, test_y):
    """
    :param params:  Naive Bayes Classifier Parameters
    :param test_x:  Test X Set
    :param test_y:  Test Y Set
    :return:        Test Data with Prediction
    """
    test_x = test_x.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
    test = test_x.copy()
    for v in params['nums']:
        test[v] = pd.cut(test[v], bins=params['bins'][v])
    classes = params['prior'].index
    for i in range(len(test)):
        post = []
        for c in classes:
            posterior = params['prior'][c]
            for var in test_x.columns:
                x = test.loc[i, var]
                posterior = posterior * params['prob'][var][c][x]
            post.append(posterior)
        test.loc[i, 'y_hat'] = classes[np.argmax(np.array(post))]
    test.loc[:, 'y'] = test_y
    return test


def model_performance(scored, prd, act):
    """
    :param scored: Scored Set with Prediction
    :param prd:    Predicted Variable Name
    :param act:    Actual Variable Name
    :return:       Accuracy Rate
    """
    correct = scored[(scored[prd] == scored[act])].index
    rt = len(correct)/len(scored)
    return "{:.2%}".format(rt)


X, y = df_dev.drop(['two_year_recid'], axis=1), df_dev['two_year_recid']

X_dev, X_val, y_dev, y_val = train_test_split(X, y, test_size=0.30, random_state=567)

pars = naive_bayes_classifier(X_dev, y_dev)

scored_dev = naive_bayes_predictor(pars, X_dev, y_dev)
scored_val = naive_bayes_predictor(pars, X_val, y_val)
scored_tst = naive_bayes_predictor(pars, df_tst.drop(['two_year_recid'], axis=1), df_tst['two_year_recid'])

rate_dev = model_performance(scored_dev, 'y_hat', 'y')
rate_val = model_performance(scored_val, 'y_hat', 'y')
rate_tst = model_performance(scored_tst, 'y_hat', 'y')

print('Accuracy Rate on Dev Sample: '  + str(rate_dev))
print('Accuracy Rate on Val Sample: '  + str(rate_val))
print('Accuracy Rate on Test Sample: ' + str(rate_tst))

