"""
ID00102866特征工程模块 - 完整版本
包含所有增强特征、动量特征和趋势特征（约100个特征）
特别针对季节性和月度模式优化
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# 月度和季节性特征（ID00102866最重要的特征）
# ============================================================================

def add_monthly_onehot_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加月份one-hot编码特征"""
    result = df.copy()
    dates = pd.DatetimeIndex(result.index)

    for month in range(1, 13):
        result[f'month_is_{month}'] = (dates.month == month).astype(int)

    return result


def add_monthly_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加月份的cyclical编码（sin/cos）"""
    result = df.copy()
    dates = pd.DatetimeIndex(result.index)

    result['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
    result['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
    result['quarter_sin'] = np.sin(2 * np.pi * dates.quarter / 4)
    result['quarter_cos'] = np.cos(2 * np.pi * dates.quarter / 4)

    return result


def add_monthly_historical_stats(df: pd.DataFrame, price_col: str, lookback_years: int = 3) -> pd.DataFrame:
    """添加每个月的历史统计特征"""
    result = df.copy()
    dates = pd.DatetimeIndex(result.index)

    # 为每个月计算历史统计
    monthly_stats = {}
    for month in range(1, 13):
        month_mask = dates.month == month
        month_data = result.loc[month_mask, price_col]

        if len(month_data) > 0:
            monthly_stats[month] = {
                'mean': month_data.mean(),
                'std': month_data.std(),
                'median': month_data.median(),
                'min': month_data.min(),
                'max': month_data.max()
            }
        else:
            monthly_stats[month] = {
                'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0
            }

    # 添加特征
    result['month_hist_mean'] = dates.month.map(lambda m: monthly_stats[m]['mean'])
    result['month_hist_std'] = dates.month.map(lambda m: monthly_stats[m]['std'])
    result['month_hist_median'] = dates.month.map(lambda m: monthly_stats[m]['median'])
    result['month_hist_min'] = dates.month.map(lambda m: monthly_stats[m]['min'])
    result['month_hist_max'] = dates.month.map(lambda m: monthly_stats[m]['max'])

    # 当前价格相对历史统计的位置
    result['price_vs_month_mean'] = (result[price_col] - result['month_hist_mean']) / (result['month_hist_std'] + 1e-10)
    result['price_vs_month_median'] = (result[price_col] - result['month_hist_median']) / (result['month_hist_median'] + 1e-10)

    return result


def add_seasonal_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """添加季节性特征"""
    result = df.copy()
    dates = pd.DatetimeIndex(result.index)

    # 季度特征
    result['quarter'] = dates.quarter
    for q in range(1, 5):
        result[f'quarter_is_{q}'] = (dates.quarter == q).astype(int)

    # 季度内的进度（0-1）
    result['quarter_progress'] = (dates.month % 3) / 3.0

    # 半年度特征
    result['half_year'] = ((dates.month - 1) // 6 + 1)
    result['half_year_is_1'] = (result['half_year'] == 1).astype(int)
    result['half_year_is_2'] = (result['half_year'] == 2).astype(int)

    # 季度历史统计
    quarterly_stats = {}
    for q in range(1, 5):
        q_mask = dates.quarter == q
        q_data = result.loc[q_mask, price_col]

        if len(q_data) > 0:
            quarterly_stats[q] = {
                'mean': q_data.mean(),
                'std': q_data.std()
            }
        else:
            quarterly_stats[q] = {'mean': 0, 'std': 0}

    result['quarter_hist_mean'] = dates.quarter.map(lambda q: quarterly_stats[q]['mean'])
    result['quarter_hist_std'] = dates.quarter.map(lambda q: quarterly_stats[q]['std'])
    result['price_vs_quarter_mean'] = (result[price_col] - result['quarter_hist_mean']) / (result['quarter_hist_std'] + 1e-10)

    return result


def add_ytd_features(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """添加年内趋势特征（Year-to-Date）"""
    result = df.copy()
    dates = pd.DatetimeIndex(result.index)

    # 年内天数
    result['day_of_year'] = dates.dayofyear
    result['days_in_year'] = pd.Series(dates.is_leap_year, index=result.index).map({True: 366, False: 365})
    result['year_progress'] = result['day_of_year'] / result['days_in_year']

    # 年初价格
    year_start_prices = {}
    for year in dates.year.unique():
        year_data = result[dates.year == year]
        if len(year_data) > 0:
            year_start_prices[year] = year_data[price_col].iloc[0]

    result['year_start_price'] = pd.Series(dates.year, index=result.index).map(year_start_prices)
    result['ytd_return'] = (result[price_col] - result['year_start_price']) / (result['year_start_price'] + 1e-10)
    result['ytd_return_pct'] = result['ytd_return'] * 100

    # 年内最高/最低价
    result['ytd_high'] = result.groupby(dates.year)[price_col].cummax()
    result['ytd_low'] = result.groupby(dates.year)[price_col].cummin()
    result['ytd_range'] = result['ytd_high'] - result['ytd_low']
    result['ytd_position'] = (result[price_col] - result['ytd_low']) / (result['ytd_range'] + 1e-10)

    return result


# ============================================================================
# 趋势特征
# ============================================================================

def add_drawdown_features(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    """添加回撤特征"""
    out = {}
    cumulative_max = series.cummax()
    out["max_drawdown_ratio"] = (series - cumulative_max) / cumulative_max.replace(0, np.nan)

    for w in windows:
        roll = series.rolling(w)
        out[f"local_drawdown_{w}"] = (series - roll.max()) / roll.max().replace(0, np.nan)
        out[f"local_recovery_{w}"] = (series - roll.min()) / roll.max().replace(0, np.nan)

    return pd.DataFrame(out, index=series.index)


def add_linear_trend_features(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    """添加线性趋势特征"""
    out = {}
    idx = np.arange(len(series), dtype=float)
    for w in windows:
        roll = series.rolling(w)

        def slope(values: np.ndarray) -> float:
            if not np.isfinite(values).all():
                return np.nan
            x = idx[: len(values)]
            A = np.vstack([x, np.ones(len(x))]).T
            try:
                m, _ = np.linalg.lstsq(A, values, rcond=None)[0]
            except np.linalg.LinAlgError:
                m = np.nan
            return m

        out[f"local_slope_{w}"] = roll.apply(slope, raw=True)
        out[f"local_accel_{w}"] = out[f"local_slope_{w}"].diff()

    return pd.DataFrame(out, index=series.index)


# ============================================================================
# 动量特征
# ============================================================================


def add_momentum_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """添加动量和方向特征"""
    result = df.copy()

    # RSI
    for window in [7, 14, 21]:
        delta = result[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        result[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        result[f'rsi_{window}_overbought'] = (result[f'rsi_{window}'] > 70).astype(int)
        result[f'rsi_{window}_oversold'] = (result[f'rsi_{window}'] < 30).astype(int)

    # MACD
    exp1 = result[price_col].ewm(span=12, adjust=False).mean()
    exp2 = result[price_col].ewm(span=26, adjust=False).mean()
    result['macd'] = exp1 - exp2
    result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
    result['macd_hist'] = result['macd'] - result['macd_signal']
    result['macd_positive'] = (result['macd'] > 0).astype(int)
    result['macd_cross_down'] = ((result['macd'].shift(1) > result['macd_signal'].shift(1)) &
                                  (result['macd'] < result['macd_signal'])).astype(int)

    # 布林带
    for window in [20, 30]:
        sma = result[price_col].rolling(window).mean()
        std = result[price_col].rolling(window).std()
        result[f'bb_upper_{window}'] = sma + 2 * std
        result[f'bb_lower_{window}'] = sma - 2 * std
        result[f'bb_middle_{window}'] = sma
        result[f'bb_position_{window}'] = (result[price_col] - result[f'bb_lower_{window}']) /                                           (result[f'bb_upper_{window}'] - result[f'bb_lower_{window}'] + 1e-10)
        result[f'bb_break_lower_{window}'] = (result[price_col] < result[f'bb_lower_{window}']).astype(int)

    # ROC
    for window in [5, 10, 20, 30]:
        result[f'roc_{window}'] = (result[price_col] - result[price_col].shift(window)) /                                   (result[price_col].shift(window) + 1e-10) * 100
        result[f'negative_momentum_{window}'] = result[f'roc_{window}'].clip(upper=0).abs()

    # 距离高点
    for window in [30, 60, 90]:
        rolling_max = result[price_col].rolling(window).max()
        result[f'distance_from_high_{window}'] = (result[price_col] - rolling_max) / (rolling_max + 1e-10) * 100
        result[f'far_from_high_{window}'] = (result[f'distance_from_high_{window}'] < -10).astype(int)

    # 连续下跌强度
    price_changes = result[price_col].diff()
    consecutive_down = []
    consecutive_up = []
    down_magnitude = []
    down_count = 0
    up_count = 0
    down_sum = 0

    for change in price_changes:
        if pd.isna(change):
            consecutive_down.append(0)
            consecutive_up.append(0)
            down_magnitude.append(0)
        elif change < 0:
            down_count += 1
            up_count = 0
            down_sum += abs(change)
            consecutive_down.append(down_count)
            consecutive_up.append(0)
            down_magnitude.append(down_sum)
        elif change > 0:
            up_count += 1
            down_count = 0
            down_sum = 0
            consecutive_down.append(0)
            consecutive_up.append(up_count)
            down_magnitude.append(0)
        else:
            consecutive_down.append(down_count)
            consecutive_up.append(up_count)
            down_magnitude.append(down_sum)

    result['consecutive_down_days'] = consecutive_down
    result['consecutive_up_days'] = consecutive_up
    result['down_magnitude'] = down_magnitude

    # 价格加速度
    for window in [5, 10]:
        result[f'price_acceleration_{window}'] = result[price_col].diff().diff(window)
        result[f'negative_acceleration_{window}'] = (result[f'price_acceleration_{window}'] < 0).astype(int)

    # 下跌信号综合
    down_signals = []
    for col in result.columns:
        if any(x in col for x in ['oversold', 'cross_down', 'break_lower', 'far_from_high', 
                                   'negative_momentum', 'negative_acceleration']):
            if col in result.columns:
                down_signals.append(col)

    if down_signals:
        result['down_signal_count'] = result[down_signals].sum(axis=1)
        result['down_signal_strength'] = result[down_signals].sum(axis=1) / len(down_signals)

    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].ffill().fillna(0)

    return result


def build_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """
    构建ID00102866的完整特征集（约100个特征）

    Args:
        df: 输入DataFrame，必须包含price列和DatetimeIndex
        price_col: 价格列名，默认'price'

    Returns:
        包含所有特征的DataFrame
    """
    result = df.copy()

    # 确保有DatetimeIndex
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("DataFrame必须有DatetimeIndex")

    # 1. 添加月度和季节性特征（ID00102866最重要的特征）
    result = add_monthly_onehot_features(result)
    result = add_monthly_cyclical_features(result)
    result = add_monthly_historical_stats(result, price_col)
    result = add_seasonal_features(result, price_col)
    result = add_ytd_features(result, price_col)

    # 2. 添加趋势特征
    price_series = result[price_col]
    drawdown_features = add_drawdown_features(price_series, windows=[7, 14, 30, 60])
    trend_features = add_linear_trend_features(price_series, windows=[7, 14, 30, 60])
    result = pd.concat([result, drawdown_features, trend_features], axis=1)

    # 3. 添加动量特征
    result = add_momentum_features(result, price_col)

    # 4. 填充NaN值
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].ffill().fillna(0)

    return result
