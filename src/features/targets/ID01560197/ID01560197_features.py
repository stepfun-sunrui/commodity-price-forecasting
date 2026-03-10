"""
ID01560197 特征构建模块
基础特征工程
"""
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """构建ID01560197的所有特征"""
    result = df.copy()
    price_series = result[price_col]
    
    # 滚动统计特征
    for window in [5, 10, 20, 30, 60]:
        result[f'ma_{window}'] = price_series.rolling(window).mean()
        result[f'std_{window}'] = price_series.rolling(window).std()
        result[f'max_{window}'] = price_series.rolling(window).max()
        result[f'min_{window}'] = price_series.rolling(window).min()
    
    # 价格变化特征
    for lag in [1, 5, 10, 20, 30]:
        result[f'return_{lag}d'] = price_series.pct_change(lag, fill_method=None)
        result[f'diff_{lag}d'] = price_series.diff(lag)
    
    # 价格位置特征
    for window in [20, 60]:
        rolling_min = price_series.rolling(window).min()
        rolling_max = price_series.rolling(window).max()
        result[f'price_position_{window}'] = (
            (price_series - rolling_min) / (rolling_max - rolling_min + 1e-8)
        )
    
    # 动量特征
    for window in [5, 10, 20]:
        momentum = price_series.diff(window)
        result[f'momentum_{window}'] = momentum
        result[f'momentum_pct_{window}'] = momentum / (price_series.shift(window) + 1e-8)
    
    # 波动率特征
    for window in [10, 20, 30]:
        returns = price_series.pct_change()
        result[f'volatility_{window}'] = returns.rolling(window).std()
    
    # 价格百分位特征
    for window in [60, 120]:
        result[f'percentile_rank_{window}'] = (
            price_series.rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
                raw=False
            )
        )
    
    # 填充NaN
    result = result.ffill().fillna(0)
    result = result.replace([np.inf, -np.inf], 0)
    
    return result
