"""
RE00035675 特征构建模块
包含88个特征：相对位置、技术指标(EMA/BB/RSI)、动量、加速度、连续性、加权平均等
"""
import pandas as pd
import numpy as np


def compute_features(daily_price_series: pd.Series) -> pd.DataFrame:
    """
    从每日价格序列计算特征

    RE00035675特点：
    - 每周公布一次数据（ffill填充是合理的）
    - 价格相对稳定，不太变化
    - 一旦变化需要及时响应

    特征设计重点：
    - 超短期窗口（3天、7天）
    - 直接价格滞后（最重要）
    - 快速响应最近变化
    - 技术指标（EMA、布林带、RSI）

    参数:
        daily_price_series: 每日价格序列 (pd.Series with DatetimeIndex)

    返回:
        features_df: 特征DataFrame，包含88个特征
    """
    features = pd.DataFrame(index=daily_price_series.index)

    # === 1. 直接滞后特征（最重要）===
    # 对于稳定序列，最近的价格是最好的预测基准
    for lag in [1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 30]:
        features[f'price_lag_{lag}'] = daily_price_series.shift(lag)

    # 最近3天的加权平均（最近的权重更大）
    features['weighted_mean_3d'] = (
        daily_price_series.shift(1) * 0.5 +
        daily_price_series.shift(2) * 0.3 +
        daily_price_series.shift(3) * 0.2
    )

    # 最近7天的加权平均
    weights_7d = [0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.1]
    features['weighted_mean_7d'] = sum(
        daily_price_series.shift(i+1) * w for i, w in enumerate(weights_7d)
    )

    # === 2. 过去3天的精确统计 ===
    features['mean_3d'] = daily_price_series.rolling(3).mean()
    features['max_3d'] = daily_price_series.rolling(3).max()
    features['min_3d'] = daily_price_series.rolling(3).min()
    features['std_3d'] = daily_price_series.rolling(3).std()

    # === 3. 过去7天（一周）的精确统计 ===
    features['mean_7d'] = daily_price_series.rolling(7).mean()
    features['max_7d'] = daily_price_series.rolling(7).max()
    features['min_7d'] = daily_price_series.rolling(7).min()
    features['std_7d'] = daily_price_series.rolling(7).std()

    # === 4. 当前价格相对于短期统计的位置 ===
    # 相对于3天
    features['vs_mean_3d'] = daily_price_series - features['mean_3d']
    features['vs_max_3d'] = daily_price_series - features['max_3d']
    features['vs_min_3d'] = daily_price_series - features['min_3d']

    # 相对于7天
    features['vs_mean_7d'] = daily_price_series - features['mean_7d']
    features['vs_max_7d'] = daily_price_series - features['max_7d']
    features['vs_min_7d'] = daily_price_series - features['min_7d']

    # === 5. 超短期变化（快速响应）===
    # 1-7天的价格变化（每天都要）
    for lag in [1, 2, 3, 4, 5, 6, 7]:
        features[f'diff_{lag}d'] = daily_price_series.diff(lag)
        features[f'return_{lag}d'] = daily_price_series.pct_change(lag, fill_method=None)

    # 最近1-3天的变化幅度（绝对值）
    features['abs_change_1d'] = daily_price_series.diff(1).abs()
    features['abs_change_2d'] = daily_price_series.diff(2).abs()
    features['abs_change_3d'] = daily_price_series.diff(3).abs()

    # 连续3天的变化趋势是否一致
    diff_1d = daily_price_series.diff(1)
    features['trend_consistency_3d'] = (
        (diff_1d * diff_1d.shift(1) > 0).astype(int) +
        (diff_1d.shift(1) * diff_1d.shift(2) > 0).astype(int)
    ) / 2.0

    # === 5.5. 短期趋势（线性）===
    features['trend_3d'] = (daily_price_series - daily_price_series.shift(3)) / 3
    features['trend_7d'] = (daily_price_series - daily_price_series.shift(7)) / 7
    features['trend_14d'] = (daily_price_series - daily_price_series.shift(14)) / 14

    # === 5.6. 价格在短期范围内的位置 ===
    range_3d = features['max_3d'] - features['min_3d']
    features['position_3d'] = (daily_price_series - features['min_3d']) / (range_3d + 1e-8)

    range_7d_val = features['max_7d'] - features['min_7d']
    features['position_7d'] = (daily_price_series - features['min_7d']) / (range_7d_val + 1e-8)

    # === 6. 技术指标 - EMA（指数移动平均）===
    features['ema_3d'] = daily_price_series.ewm(span=3, adjust=False).mean()
    features['ema_7d'] = daily_price_series.ewm(span=7, adjust=False).mean()
    features['ema_14d'] = daily_price_series.ewm(span=14, adjust=False).mean()

    # 价格相对于EMA的位置
    features['vs_ema_3d'] = daily_price_series - features['ema_3d']
    features['vs_ema_7d'] = daily_price_series - features['ema_7d']
    features['vs_ema_14d'] = daily_price_series - features['ema_14d']

    # === 7. 技术指标 - 布林带（Bollinger Bands）===
    # 7天布林带
    bb_window = 7
    bb_std = 2
    bb_mean = daily_price_series.rolling(bb_window).mean()
    bb_std_val = daily_price_series.rolling(bb_window).std()
    features['bb_upper_7d'] = bb_mean + bb_std * bb_std_val
    features['bb_lower_7d'] = bb_mean - bb_std * bb_std_val
    features['bb_width_7d'] = features['bb_upper_7d'] - features['bb_lower_7d']

    # 价格在布林带中的位置（0-1之间）
    features['bb_position_7d'] = (
        (daily_price_series - features['bb_lower_7d']) /
        (features['bb_width_7d'] + 1e-8)
    )

    # === 8. 技术指标 - RSI（相对强弱指标）===
    # 7天RSI
    delta = daily_price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / (loss + 1e-8)
    features['rsi_7d'] = 100 - (100 / (1 + rs))

    # 14天RSI
    gain_14 = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss_14 = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs_14 = gain_14 / (loss_14 + 1e-8)
    features['rsi_14d'] = 100 - (100 / (1 + rs_14))

    # === 9. 动量特征 ===
    # 价格动量（变化速度）
    features['momentum_3d'] = daily_price_series - daily_price_series.shift(3)
    features['momentum_7d'] = daily_price_series - daily_price_series.shift(7)
    features['momentum_14d'] = daily_price_series - daily_price_series.shift(14)

    # 动量变化率
    features['momentum_change_3d'] = features['momentum_3d'].diff(1)
    features['momentum_change_7d'] = features['momentum_7d'].diff(1)

    # === 10. 加速度特征 ===
    # 价格变化的变化（二阶导数）
    diff_1d = daily_price_series.diff(1)
    features['accel_1d'] = diff_1d.diff(1)
    features['accel_2d'] = daily_price_series.diff(2).diff(2)
    features['accel_3d'] = daily_price_series.diff(3).diff(3)
    features['accel_7d'] = daily_price_series.diff(7).diff(7)

    # === 10.5. 最近一周的极值特征 ===
    rolling_7d_max = daily_price_series.rolling(7).max()
    rolling_7d_min = daily_price_series.rolling(7).min()

    # 当前价格距离7天最高/最低的天数
    features['days_since_7d_high'] = 0.0
    features['days_since_7d_low'] = 0.0

    for i in range(1, 8):
        is_max = (daily_price_series.shift(i) == rolling_7d_max).astype(int)
        is_min = (daily_price_series.shift(i) == rolling_7d_min).astype(int)
        features['days_since_7d_high'] += is_max * i
        features['days_since_7d_low'] += is_min * i

    # 是否是7天最高/最低
    features['is_7d_high'] = (daily_price_series == rolling_7d_max).astype(int)
    features['is_7d_low'] = (daily_price_series == rolling_7d_min).astype(int)

    # === 11. 波动率和范围 ===
    # 短期波动率
    returns = daily_price_series.pct_change(fill_method=None)
    features['volatility_3d'] = returns.rolling(3).std()
    features['volatility_5d'] = returns.rolling(5).std()
    features['volatility_7d'] = returns.rolling(7).std()
    features['volatility_14d'] = returns.rolling(14).std()

    # 波动率比率（短期vs中期）
    features['vol_ratio_3d_14d'] = features['volatility_3d'] / (features['volatility_14d'] + 1e-8)

    # 变异系数（CV = std/mean）
    features['cv_3d'] = features['std_3d'] / (features['mean_3d'] + 1e-8)
    features['cv_7d'] = features['std_7d'] / (features['mean_7d'] + 1e-8)

    # 价格范围
    range_7d = features['max_7d'] - features['min_7d']
    features['range_7d'] = range_7d

    # 价格范围占比
    features['range_ratio_3d'] = (features['max_3d'] - features['min_3d']) / (features['mean_3d'] + 1e-8)
    features['range_ratio_7d'] = range_7d / (features['mean_7d'] + 1e-8)

    # === 12. 最近价格的连续性 ===
    # 最近3天价格是否连续上涨/下跌
    features['continuous_up_3d'] = (
        ((daily_price_series > daily_price_series.shift(1)) &
         (daily_price_series.shift(1) > daily_price_series.shift(2)) &
         (daily_price_series.shift(2) > daily_price_series.shift(3)))
    ).astype(int)

    features['continuous_down_3d'] = (
        ((daily_price_series < daily_price_series.shift(1)) &
         (daily_price_series.shift(1) < daily_price_series.shift(2)) &
         (daily_price_series.shift(2) < daily_price_series.shift(3)))
    ).astype(int)

    # === 13. 中期参考（较小权重）===
    features['ma_14d'] = daily_price_series.rolling(14).mean()
    features['ma_30d'] = daily_price_series.rolling(30).mean()
    features['std_14d'] = daily_price_series.rolling(14).std()

    # 相对于14天、30天均值的偏离
    features['vs_ma_14d'] = daily_price_series - features['ma_14d']
    features['vs_ma_30d'] = daily_price_series - features['ma_30d']

    # 14天变异系数
    features['cv_14d'] = features['std_14d'] / (features['ma_14d'] + 1e-8)

    # === NaN处理 ===
    # 使用前向填充，然后用0填充剩余的NaN
    features = features.ffill().fillna(0)

    return features


def build_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """
    构建RE00035675的所有特征

    这是主接口函数，内部调用compute_features()

    参数:
        df: 包含价格列的DataFrame
        price_col: 价格列名，默认为'price'

    返回:
        包含原始数据和所有特征的DataFrame
    """
    result = df.copy()

    # 调用compute_features计算所有特征
    features = compute_features(result[price_col])

    # 合并特征到结果DataFrame
    for col in features.columns:
        result[col] = features[col]

    return result
