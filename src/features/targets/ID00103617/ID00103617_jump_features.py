"""
跳跃预测特征模块

针对ID00103617的阶梯式价格序列设计的特征：
- 88%时间价格保持不变
- 突发性跳跃（非渐进式）
- 跳跃前往往经历长期稳定

核心思想：长期稳定 → 跳跃风险增加
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
        hist_vol = price_series.pct_change().rolling(lookback, min_periods=1).std()
        recent_vol = price_series.pct_change().rolling(7, min_periods=1).std()

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
        returns = price_series.pct_change()
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

    # 统计新增特征数
    new_feature_count = sum(
        1 for c in result.columns
        if c.startswith(('days_stable', 'vol_comp', 'at_bound', 'boundary_dur',
                        'days_since', 'jump_', 'trend_consist'))
    )

    print(f"[OK] Added jump prediction features: {new_feature_count}")

    return result
