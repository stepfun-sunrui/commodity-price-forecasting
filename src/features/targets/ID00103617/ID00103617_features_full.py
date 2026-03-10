"""
特征工程模块 - 为ID00103617生成增强特征
针对日度数据、有突发跳跃的序列，需要检测跳跃模式和趋势变化
"""
import pandas as pd
import numpy as np


def compute_features(daily_price_series: pd.Series, include_jump: bool = True) -> pd.DataFrame:
    """
    从每日价格序列计算特征

    ID00103617特点：
    - 日度数据，密度高
    - 存在突发性价格跳跃（如2025-07-25: +90）
    - 需要预测跳跃后的新价格水平

    Parameters:
    -----------
    daily_price_series : pd.Series
        每日价格序列
    include_jump : bool, default=True
        是否包含跳跃预测特征

    特征设计重点：
    1. 历史跳跃模式检测
    2. 价格相对位置（分位数）
    3. 多时间尺度趋势
    4. 波动率变化检测
    5. 区间突破特征

    参数:
        daily_price_series: 每日价格序列 (pd.Series with DatetimeIndex)

    返回:
        features_df: 特征DataFrame
    """
    features = pd.DataFrame(index=daily_price_series.index)

    # === 1. 基础滞后特征 ===
    for lag in [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90]:
        features[f'price_lag_{lag}'] = daily_price_series.shift(lag)

    # === 2. 价格变化特征 ===
    # 日度变化
    for lag in [1, 2, 3, 5, 7, 14]:
        features[f'diff_{lag}d'] = daily_price_series.diff(lag)
        features[f'return_{lag}d'] = daily_price_series.pct_change(lag)
        features[f'abs_change_{lag}d'] = daily_price_series.diff(lag).abs()

    # === 3. 价格跳跃检测特征 ===
    # 检测历史上的大幅跳跃（>5%）
    returns_1d = daily_price_series.pct_change()

    # 最近N天内是否有大跳跃
    large_jump_threshold = 0.05  # 5%
    for window in [3, 7, 14, 30]:
        # 向上跳跃
        up_jumps = (returns_1d > large_jump_threshold).astype(int)
        features[f'has_large_up_jump_{window}d'] = up_jumps.rolling(window).max()
        features[f'days_since_up_jump_{window}d'] = features.index.to_series().diff().dt.days

        # 向下跳跃
        down_jumps = (returns_1d < -large_jump_threshold).astype(int)
        features[f'has_large_down_jump_{window}d'] = down_jumps.rolling(window).max()

        # 最近N天最大单日涨跌幅
        features[f'max_return_{window}d'] = returns_1d.rolling(window).max()
        features[f'min_return_{window}d'] = returns_1d.rolling(window).min()

    # 连续上涨/下跌天数
    price_change = daily_price_series.diff()
    features['consecutive_up_days'] = (
        (price_change > 0).groupby((price_change <= 0).cumsum()).cumsum()
    )
    features['consecutive_down_days'] = (
        (price_change < 0).groupby((price_change >= 0).cumsum()).cumsum()
    )

    # === 4. 价格分位数特征（相对历史位置）===
    for window in [30, 60, 90, 180, 365]:
        rolling_min = daily_price_series.rolling(window).min()
        rolling_max = daily_price_series.rolling(window).max()
        rolling_range = rolling_max - rolling_min

        # 价格在区间中的位置（0-1）
        features[f'price_position_{window}d'] = (
            (daily_price_series - rolling_min) / (rolling_range + 1e-8)
        )

        # 距离区间上下边界的距离
        features[f'dist_to_high_{window}d'] = rolling_max - daily_price_series
        features[f'dist_to_low_{window}d'] = daily_price_series - rolling_min

        # 是否创新高/新低
        features[f'is_new_high_{window}d'] = (daily_price_series == rolling_max).astype(int)
        features[f'is_new_low_{window}d'] = (daily_price_series == rolling_min).astype(int)

    # === 5. 均值回归特征 ===
    for window in [7, 14, 30, 60, 90]:
        rolling_mean = daily_price_series.rolling(window).mean()

        # 偏离均值的程度
        features[f'deviation_from_ma_{window}d'] = daily_price_series - rolling_mean
        features[f'deviation_pct_{window}d'] = (
            (daily_price_series - rolling_mean) / (rolling_mean + 1e-8)
        )

        # 均值的变化率（趋势）
        features[f'ma_change_{window}d'] = rolling_mean.diff(7)
        features[f'ma_slope_{window}d'] = rolling_mean.diff(7) / 7.0

    # === 6. 多时间尺度统计特征 ===
    for window in [3, 7, 14, 30, 60, 90]:
        features[f'mean_{window}d'] = daily_price_series.rolling(window).mean()
        features[f'std_{window}d'] = daily_price_series.rolling(window).std()
        features[f'max_{window}d'] = daily_price_series.rolling(window).max()
        features[f'min_{window}d'] = daily_price_series.rolling(window).min()
        features[f'median_{window}d'] = daily_price_series.rolling(window).median()

        # 变异系数（相对波动性）
        features[f'cv_{window}d'] = (
            features[f'std_{window}d'] / (features[f'mean_{window}d'] + 1e-8)
        )

    # === 7. 波动率特征及其变化 ===
    returns = daily_price_series.pct_change()

    for window in [3, 7, 14, 30, 60]:
        volatility = returns.rolling(window).std()
        features[f'volatility_{window}d'] = volatility

        # 波动率的变化（波动率突变可能预示价格跳跃）
        features[f'volatility_change_{window}d'] = volatility.diff(3)
        features[f'volatility_ratio_{window}d'] = volatility / (volatility.shift(window) + 1e-8)

    # 波动率相对比率
    features['vol_ratio_7d_30d'] = (
        features['volatility_7d'] / (features['volatility_30d'] + 1e-8)
    )
    features['vol_ratio_14d_60d'] = (
        features['volatility_14d'] / (features['volatility_60d'] + 1e-8)
    )

    # === 8. 趋势指标 ===
    # 多时间尺度的趋势一致性
    for short, long in [(7, 30), (14, 60), (30, 90)]:
        ma_short = daily_price_series.rolling(short).mean()
        ma_long = daily_price_series.rolling(long).mean()

        features[f'ma_cross_{short}_{long}'] = (ma_short > ma_long).astype(int)
        features[f'ma_divergence_{short}_{long}'] = ma_short - ma_long
        features[f'ma_divergence_pct_{short}_{long}'] = (
            (ma_short - ma_long) / (ma_long + 1e-8)
        )

    # 趋势强度（线性回归斜率）
    for window in [7, 14, 30, 60]:
        slopes = []
        for i in range(len(daily_price_series)):
            if i < window:
                slopes.append(np.nan)
            else:
                y = daily_price_series.iloc[i-window:i].values
                x = np.arange(window)
                if len(y) == window and not np.any(np.isnan(y)):
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        features[f'trend_slope_{window}d'] = slopes

    # === 9. 加权移动平均 ===
    # EMA（指数移动平均）对最近价格更敏感
    for span in [5, 10, 20, 40]:
        features[f'ema_{span}d'] = daily_price_series.ewm(span=span).mean()
        features[f'price_to_ema_{span}d'] = daily_price_series - features[f'ema_{span}d']

    # === 10. 价格加速度（二阶差分）===
    for lag in [1, 3, 7]:
        features[f'acceleration_{lag}d'] = daily_price_series.diff(lag).diff(lag)

    # === 11. 区间突破特征 ===
    # 布林带
    for window in [20, 40]:
        ma = daily_price_series.rolling(window).mean()
        std = daily_price_series.rolling(window).std()

        features[f'bollinger_upper_{window}d'] = ma + 2 * std
        features[f'bollinger_lower_{window}d'] = ma - 2 * std
        features[f'bollinger_position_{window}d'] = (
            (daily_price_series - ma) / (std + 1e-8)
        )

        # 是否突破布林带
        features[f'above_bollinger_{window}d'] = (
            daily_price_series > ma + 2 * std
        ).astype(int)
        features[f'below_bollinger_{window}d'] = (
            daily_price_series < ma - 2 * std
        ).astype(int)

    # === 12. 稳定性特征 ===
    # 最近N天价格是否稳定（标准差很小）
    for window in [7, 14, 30]:
        std = daily_price_series.rolling(window).std()
        mean = daily_price_series.rolling(window).mean()
        cv = std / (mean + 1e-8)

        # 低波动标志
        features[f'is_stable_{window}d'] = (cv < 0.02).astype(int)  # CV<2%视为稳定
        features[f'stability_score_{window}d'] = 1.0 / (cv + 1e-8)

    # === 13. 价格水平切换检测 ===
    # 检测价格是否从一个水平跳到另一个水平并维持
    for window in [7, 14]:
        # 最近window天的价格范围
        recent_std = daily_price_series.rolling(window).std()

        # 价格与window天前的差异
        price_shift = daily_price_series.diff(window).abs()

        # 如果差异大但最近稳定，说明发生了水平切换
        features[f'level_switch_{window}d'] = (
            (price_shift > 50) & (recent_std < 10)
        ).astype(int)

    # 集成跳跃预测特征（优化核心）
    if include_jump:
        try:
            from jump_features import add_jump_prediction_features
            features = add_jump_prediction_features(features, daily_price_series)
        except ImportError:
            print("[WARNING] jump_features.py not found, skipping jump features")
        except Exception as e:
            print(f"[WARNING] Failed to add jump features: {e}")

    return features


if __name__ == "__main__":
    # 简单测试
    dates = pd.date_range('2020-01-01', '2025-10-30', freq='D')
    prices = pd.Series(1000 + np.random.randn(len(dates)).cumsum() * 10, index=dates)

    features = compute_features(prices)
    print(f"生成特征数: {len(features.columns)}")
    print(f"特征名称示例: {list(features.columns[:10])}")
