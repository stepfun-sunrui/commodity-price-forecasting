"""
特征工程模块 - 为ID00103568生成基础特征
针对每周公布、价格稳定、需要快速响应变化的序列
"""
import pandas as pd
import numpy as np


def compute_features(daily_price_series: pd.Series) -> pd.DataFrame:
    """
    从每日价格序列计算特征

    ID00103568特点：
    - 每周公布一次数据（ffill填充是合理的）
    - 价格相对稳定，不太变化
    - 一旦变化需要及时响应

    特征设计重点：
    - 超短期窗口（3天、7天）
    - 直接价格滞后（最重要）
    - 快速响应最近变化

    参数:
        daily_price_series: 每日价格序列 (pd.Series with DatetimeIndex)

    返回:
        features_df: 特征DataFrame
    """
    features = pd.DataFrame(index=daily_price_series.index)

    # === 1. 直接滞后特征（最重要）===
    # 对于稳定序列，最近的价格是最好的预测基准
    # 增加更密集的短期滞后（每天都要）
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

    # === 3.5 过去14天和30天的精确统计 ===
    features['mean_14d'] = daily_price_series.rolling(14).mean()
    features['max_14d'] = daily_price_series.rolling(14).max()
    features['min_14d'] = daily_price_series.rolling(14).min()
    features['std_14d'] = daily_price_series.rolling(14).std()

    features['mean_30d'] = daily_price_series.rolling(30).mean()
    features['max_30d'] = daily_price_series.rolling(30).max()
    features['min_30d'] = daily_price_series.rolling(30).min()
    features['std_30d'] = daily_price_series.rolling(30).std()

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

    # === 6. 加速度（变化的变化）===
    # 1天变化的变化
    diff_1d = daily_price_series.diff(1)
    features['accel_1d'] = diff_1d.diff(1)
    features['accel_2d'] = daily_price_series.diff(2).diff(2)
    # 3天变化的变化
    diff_3d = daily_price_series.diff(3)
    features['accel_3d'] = diff_3d.diff(3)
    diff_7d = daily_price_series.diff(7)
    features['accel_7d'] = diff_7d.diff(7)

    # === 7. 动量指标 ===
    # 3天动量
    features['momentum_3d'] = daily_price_series - daily_price_series.shift(3)
    # 7天动量
    features['momentum_7d'] = daily_price_series - daily_price_series.shift(7)

    # === 8. 短期趋势（平均每日变化）===
    features['trend_3d'] = (daily_price_series - daily_price_series.shift(3)) / 3
    features['trend_7d'] = (daily_price_series - daily_price_series.shift(7)) / 7
    features['trend_14d'] = (daily_price_series - daily_price_series.shift(14)) / 14

    # === 9. 波动率（变化的标准差）===
    returns = daily_price_series.pct_change(fill_method=None)
    features['volatility_3d'] = returns.rolling(3).std()
    features['volatility_5d'] = returns.rolling(5).std()
    features['volatility_7d'] = returns.rolling(7).std()
    features['volatility_14d'] = returns.rolling(14).std()

    # 最近3天波动率相对于14天波动率的比值（检测波动变化）
    vol_14d = returns.rolling(14).std()
    features['vol_ratio_3d_14d'] = features['volatility_3d'] / (vol_14d + 1e-8)

    # === 10. 价格范围 ===
    range_3d = features['max_3d'] - features['min_3d']
    range_7d = features['max_7d'] - features['min_7d']
    features['range_3d'] = range_3d
    features['range_7d'] = range_7d

    # 价格在短期范围内的位置
    features['position_3d'] = (daily_price_series - features['min_3d']) / (range_3d + 1e-8)
    features['position_7d'] = (daily_price_series - features['min_7d']) / (range_7d + 1e-8)

    # === 11. 变化率的变化（二阶导数）===
    # 1天变化率的变化
    features['return_change_1d'] = features['return_1d'].diff(1)
    # 3天变化率的变化
    features['return_change_3d'] = features['return_3d'].diff(1)

    # 价格范围占比
    features['range_ratio_3d'] = (features['max_3d'] - features['min_3d']) / (features['mean_3d'] + 1e-8)
    features['range_ratio_7d'] = range_7d / (features['mean_7d'] + 1e-8)

    # === 11.5 最近一周的极值特征 ===
    # 记录最近7天内的最高价和最低价
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

    # 当前价格是否是7天内的新高/新低
    features['is_7d_high'] = (daily_price_series == rolling_7d_max).astype(int)
    features['is_7d_low'] = (daily_price_series == rolling_7d_min).astype(int)

    # === 11.6 稳定性指标 ===
    # 变异系数（相对波动）
    features['cv_3d'] = features['std_3d'] / (features['mean_3d'] + 1e-8)
    features['cv_7d'] = features['std_7d'] / (features['mean_7d'] + 1e-8)
    features['cv_14d'] = daily_price_series.rolling(14).std() / (daily_price_series.rolling(14).mean() + 1e-8)

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

    # 相对于14天、30天均值的偏离
    features['vs_ma_14d'] = daily_price_series - features['ma_14d']
    features['vs_ma_30d'] = daily_price_series - features['ma_30d']

    return features


def build_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """
    构建ID00103568的所有特征

    参数:
        df: 包含价格列的DataFrame
        price_col: 价格列名，默认为'price'

    返回:
        包含所有特征的DataFrame
    """
    result = df.copy()

    # 调用compute_features计算特征
    features = compute_features(result[price_col])

    # 合并特征到结果DataFrame
    for col in features.columns:
        result[col] = features[col]

    # NaN处理：先前向填充，再用0填充剩余的NaN
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].ffill().fillna(0)

    return result
