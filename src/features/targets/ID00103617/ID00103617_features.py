"""
特征工程模块 - ID00103617
针对日度数据、有突发跳跃的序列，需要检测跳跃模式和趋势变化
包含完整的125个特征
"""
import pandas as pd
import numpy as np


def add_jump_prediction_features(df: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
    """
    添加跳跃预测特征

    Parameters:
    -----------
    df : pd.DataFrame
        现有特征DataFrame
    price_series : pd.Series
        价格序列（索引为DatetimeIndex）

    Returns:
    --------
    pd.DataFrame
        添加了跳跃预测特征的DataFrame
    """
    result = df.copy()

    # 1. 稳定期持续天数（关键特征）
    # 价格变化<5视为稳定
    price_changes = price_series.diff().abs()
    days_stable = []
    count = 0

    for change in price_changes:
        if pd.isna(change) or change < 5:
            count += 1
        else:
            count = 0
        days_stable.append(count)

    result['days_stable'] = days_stable
    result['days_stable_norm'] = np.clip(np.array(days_stable) / 30.0, 0, 1)

    # 2. 稳定期长度在历史中的分位数
    # 当前稳定期在过去90天中的排名（越高说明稳定时间越长，越可能跳跃）
    result['stable_percentile'] = result['days_stable'].rolling(90, min_periods=1).rank(pct=True)

    # 3. 波动率压缩（低波动预示大波动）
    # 短期波动率 vs 长期波动率的比值
    for lookback in [30, 60]:
        hist_vol = price_series.pct_change(fill_method=None).rolling(lookback, min_periods=1).std()
        recent_vol = price_series.pct_change(fill_method=None).rolling(7, min_periods=1).std()

        result[f'vol_compression_{lookback}d'] = recent_vol / (hist_vol + 1e-8)

    # 4. 价格在区间边界（接近历史高/低点，可能突破）
    for lookback in [60, 90, 180]:
        rolling_min = price_series.rolling(lookback, min_periods=1).min()
        rolling_max = price_series.rolling(lookback, min_periods=1).max()
        price_range = rolling_max - rolling_min

        # 价格在区间中的相对位置 [0, 1]
        price_position = (price_series - rolling_min) / (price_range + 1e-8)

        # 接近边界（<10% 或 >90%）
        result[f'at_boundary_{lookback}d'] = (
            (price_position < 0.1) | (price_position > 0.9)
        ).astype(int)

        # 连续边界时长（可能是突破前的整固）
        result[f'boundary_duration_{lookback}d'] = (
            result[f'at_boundary_{lookback}d']
            .groupby((result[f'at_boundary_{lookback}d'] != result[f'at_boundary_{lookback}d'].shift()).cumsum())
            .cumsum()
        )

    # 5. 历史跳跃间隔统计
    # 距离上次大跳跃的天数
    large_jumps = price_changes > 50  # 定义大跳跃：>50点
    days_since_jump = []
    last_jump_idx = None

    for i, is_jump in enumerate(large_jumps):
        if is_jump:
            last_jump_idx = i
            days_since_jump.append(0)
        else:
            if last_jump_idx is None:
                days_since_jump.append(i)  # 从序列开始算起
            else:
                days_since_jump.append(i - last_jump_idx)

    result['days_since_last_jump'] = days_since_jump
    result['days_since_jump_norm'] = np.clip(np.array(days_since_jump) / 60.0, 0, 1)

    # 6. 跳跃频率（最近N天的跳跃次数）
    for lookback in [90, 180, 365]:
        result[f'jump_frequency_{lookback}d'] = large_jumps.rolling(lookback, min_periods=1).sum()

    # 7. 价格趋势一致性（如果长期单边，可能积累反转压力）
    for lookback in [30, 60]:
        returns = price_series.pct_change(fill_method=None)
        up_days = (returns > 0).rolling(lookback, min_periods=1).sum()
        down_days = (returns < 0).rolling(lookback, min_periods=1).sum()

        result[f'trend_consistency_{lookback}d'] = (up_days - down_days) / lookback

    # 8. 综合跳跃风险评分
    # 多个信号的组合
    signals = []

    # 信号1：长期稳定（>20天无大变化）
    signals.append((result['days_stable'] > 20).astype(int))

    # 信号2：稳定期在历史中排名高（>80%分位）
    signals.append((result['stable_percentile'] > 0.8).astype(int))

    # 信号3：波动率压缩（<30%）
    signals.append((result['vol_compression_30d'] < 0.3).astype(int))

    # 信号4：接近历史边界
    signals.append(result['at_boundary_60d'])

    # 信号5：距离上次跳跃时间较长（>30天）
    signals.append((result['days_since_last_jump'] > 30).astype(int))

    # 合并信号
    signals_df = pd.concat(signals, axis=1, keys=[f'signal_{i}' for i in range(len(signals))])
    result['jump_risk_score'] = signals_df.sum(axis=1)
    result['jump_risk_high'] = (result['jump_risk_score'] >= 3).astype(int)

    return result


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
        features[f'return_{lag}d'] = daily_price_series.pct_change(lag, fill_method=None)
        features[f'abs_change_{lag}d'] = daily_price_series.diff(lag).abs()

    # === 3. 价格跳跃检测特征 ===
    # 检测历史上的大幅跳跃（>5%）
    returns_1d = daily_price_series.pct_change(fill_method=None)

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
    # 价格在历史窗口中的分位数排名
    for window in [30, 60, 90, 180, 365]:
        features[f'price_percentile_{window}d'] = (
            daily_price_series.rolling(window, min_periods=1).rank(pct=True)
        )

    # === 5. 滚动统计特征 ===
    for window in [7, 14, 30, 60, 90]:
        # 均值
        features[f'ma_{window}'] = daily_price_series.rolling(window).mean()

        # 标准差（波动率）
        features[f'std_{window}'] = daily_price_series.rolling(window).std()

        # 变异系数（相对波动率）
        features[f'cv_{window}'] = features[f'std_{window}'] / (features[f'ma_{window}'] + 1e-8)

        # 最大值、最小值
        features[f'max_{window}'] = daily_price_series.rolling(window).max()
        features[f'min_{window}'] = daily_price_series.rolling(window).min()

        # 价格相对区间位置
        price_range = features[f'max_{window}'] - features[f'min_{window}']
        features[f'price_position_{window}'] = (
            (daily_price_series - features[f'min_{window}']) / (price_range + 1e-8)
        )

    # === 6. 价格与均线的关系 ===
    for window in [7, 14, 30, 60]:
        ma = daily_price_series.rolling(window).mean()
        features[f'price_to_ma_{window}'] = daily_price_series / (ma + 1e-8)
        features[f'price_minus_ma_{window}'] = daily_price_series - ma

    # === 7. 趋势强度特征 ===
    for window in [14, 30, 60]:
        # 线性回归斜率（趋势方向和强度）
        def calc_slope(series):
            if len(series) < 2 or series.isna().all():
                return np.nan
            x = np.arange(len(series))
            y = series.values
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                return np.nan
            return np.polyfit(x[valid], y[valid], 1)[0]

        features[f'trend_slope_{window}'] = (
            daily_price_series.rolling(window).apply(calc_slope, raw=False)
        )

    # === 8. 波动率变化特征 ===
    # 短期波动率 vs 长期波动率
    short_vol = daily_price_series.pct_change(fill_method=None).rolling(7).std()
    for long_window in [30, 60, 90]:
        long_vol = daily_price_series.pct_change(fill_method=None).rolling(long_window).std()
        features[f'vol_ratio_7_{long_window}'] = short_vol / (long_vol + 1e-8)

    # === 9. 动量特征 ===
    for window in [7, 14, 30]:
        # ROC (Rate of Change)
        features[f'roc_{window}'] = (
            (daily_price_series - daily_price_series.shift(window)) /
            (daily_price_series.shift(window) + 1e-8)
        )

    # === 10. 价格加速度（二阶导数）===
    for window in [3, 7, 14]:
        # 一阶差分的差分
        features[f'acceleration_{window}'] = daily_price_series.diff(window).diff()

    # === 11. 区间突破特征 ===
    for window in [30, 60, 90]:
        rolling_max = daily_price_series.rolling(window).max()
        rolling_min = daily_price_series.rolling(window).min()

        # 突破历史高点
        features[f'breakout_high_{window}'] = (
            daily_price_series > rolling_max.shift(1)
        ).astype(int)

        # 跌破历史低点
        features[f'breakout_low_{window}'] = (
            daily_price_series < rolling_min.shift(1)
        ).astype(int)

    # === 12. 价格稳定性特征 ===
    for window in [7, 14, 30]:
        # 价格变化的标准差（衡量稳定性）
        features[f'change_stability_{window}'] = (
            daily_price_series.diff().rolling(window).std()
        )

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
        features = add_jump_prediction_features(features, daily_price_series)

    return features


def build_features(df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
    """
    构建ID00103617的所有特征（统一接口）

    Parameters:
    -----------
    df : pd.DataFrame
        包含价格列的DataFrame
    price_col : str, default='price'
        价格列名

    Returns:
    --------
    pd.DataFrame
        包含所有特征的DataFrame
    """
    price_series = df[price_col]

    # 调用compute_features生成所有特征
    features = compute_features(price_series, include_jump=True)

    # 合并原始数据
    result = pd.concat([df, features], axis=1)

    # NaN处理：先前向填充，再用0填充剩余NaN
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].ffill().fillna(0)

    return result

