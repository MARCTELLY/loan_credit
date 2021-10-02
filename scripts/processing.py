import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, SelectFwe


class Error(Exception):
    pass


class SelectorError(Error):
    """
    This raise error for selector
    """

    def __init__(self, msg):
        self.msg = msg


class QueryError(Error):
    """
    This handle error for query
    """

    def __int__(self, msg):
        self.msg = msg


class SomeError(Error):
    """
    This handle error for query
    """

    def __int__(self, msg):
        self.msg = msg


def get_na_percentage(df: 'pd.DataFrame') -> 'pd.Series':
    """
    this function return percentage of na in each columns
    """
    print(df.shape)
    return df.isna().sum().apply(lambda x: x*100/df.shape[0])


def select_column(df: 'pd.DataFrame', na_threshold: int = 50) -> 'pd.DataFrame':
    """
    select in dataframe column with an threshold of NAN values
    """
    return df[get_na_percentage(df).index[get_na_percentage(df).index <= na_threshold]]


def split_dataframe_by_dtype(df: 'pd.DataFrame') -> list:
    """
    this function split dataframe in 2
    """
    return df.select_dtypes(exclude="object"), df.select_dtypes(include="object")


def change_y_encoding(value: str) -> str:
    """
    This function tranform value of y 
    """
    if value in ['Fully Paid', 'Current']:
        return "paid"
    else:
        return "unpaid"


def transform_y(df: 'pd.Series') -> 'pd.Series':
    """
    Applying tranformation of y
    """
    return df.apply(change_y_encoding)


def selection_variable(X: 'pd.DataFrame', Y: 'pd.Series', selector: str = "SelectFwe",
                       stat_test: str = 'f_classif', alpha: float = 0.001, k: int = 20) -> 'pd.DataFrame':
    """
    This function select best variable based on some scoring 
    """
    select = None
    if selector == "SelectFwe" and stat_test is not None:
        select = selector(stat_test, alpha=alpha)
        res = select.fit_transform(X, Y)
        return f"{alpha} : alpha", X[X.columns[select.get_support()]]
    elif selector == "SelectKBest" and stat_test is not None:
        select = selector(stat_test, k=k)
        res = select.fit_transform(X, Y)
        return f"k : {k}", X[X.columns[select.get_support()]]

    else:
        raise SelectorError("There is some error with your selector")


def transform_categorical(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    This function tranform object data to integer
    """
    return df.apply(lambda x: pd.factorize(x)[0])


def numeric_fillna(df: 'pd.DataFrame', method: str, value=None):

    try:
        if method is None:
            return df.apply(lambda x: x.fillna(x.method()))
        else:
            return df.apply(lambda x: x.fillna(value))
    except:
        raise SomeError("there are error in with function argurment")


def category_fillna(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Fill nan value with median in categorical data
    """
    return df.apply(lambda x: x.fillna(x.median()))
