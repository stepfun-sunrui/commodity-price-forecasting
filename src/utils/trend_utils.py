"""
趋势分析工具模块
用于为预测结果添加趋势分析列
"""
import pandas as pd
import numpy as np


def get_trend(value_diff):
    """
    根据价格差值判断趋势

    Args:
        value_diff: 价格差值（目标价格 - 起点价格）

    Returns:
        str: '上升', '下降', '不变'
    """
    if value_diff > 20:
        return '上升'
    elif value_diff < -20:
        return '下降'
    else:
        return '不变'


def add_trend_columns(df_predictions, real_price_series):
    """
    为predictions DataFrame添加趋势列

    Args:
        df_predictions: 包含target_date, origin_date, actual, pred列的DataFrame
        real_price_series: 真实价格的Series，index为日期（DatetimeIndex）

    Returns:
        DataFrame: 添加了actual_trend和pred_trend列的DataFrame
    """
    df = df_predictions.copy()

    # 确保日期列是datetime类型
    df['origin_date'] = pd.to_datetime(df['origin_date'])

    # 确保real_price_series的index也是datetime
    if not isinstance(real_price_series.index, pd.DatetimeIndex):
        real_price_series.index = pd.to_datetime(real_price_series.index)

    # 标准化日期（只保留日期部分，去除时间）
    df['origin_date_normalized'] = df['origin_date'].dt.normalize()
    price_index_normalized = real_price_series.index.normalize()

    # 创建日期到价格的映射字典（使用标准化后的日期）
    price_dict = dict(zip(price_index_normalized, real_price_series.values))

    # 获取origin_date的价格
    df['origin_price'] = df['origin_date_normalized'].map(price_dict)

    # 如果有缺失值，尝试向前填充（使用最近的交易日价格）
    if df['origin_price'].isna().any():
        print(f"  警告: {df['origin_price'].isna().sum()} 条记录的origin_date没有直接匹配的价格，尝试使用最近的历史价格")

        # 对每个缺失的日期，找到最近的历史价格
        for idx in df[df['origin_price'].isna()].index:
            target_date = df.loc[idx, 'origin_date_normalized']
            # 找到小于等于target_date的最近日期
            valid_dates = price_index_normalized[price_index_normalized <= target_date]
            if len(valid_dates) > 0:
                nearest_date = valid_dates.max()
                df.loc[idx, 'origin_price'] = price_dict[nearest_date]

    # 计算价格差值
    df['actual_diff'] = df['actual'] - df['origin_price']
    df['pred_diff'] = df['pred'] - df['origin_price']

    # 计算趋势
    df['actual_trend'] = df['actual_diff'].apply(lambda x: get_trend(x) if pd.notna(x) else np.nan)
    df['pred_trend'] = df['pred_diff'].apply(lambda x: get_trend(x) if pd.notna(x) else np.nan)

    # 删除中间列
    df = df.drop(columns=['origin_price', 'actual_diff', 'pred_diff', 'origin_date_normalized'])

    return df


def calculate_trend_accuracy(df_predictions):
    """
    计算每月的趋势准确率

    Args:
        df_predictions: 必须包含month, actual_trend, pred_trend列

    Returns:
        DataFrame: 包含month和trend_accuracy列
    """
    if 'month' not in df_predictions.columns:
        raise ValueError("df_predictions must have 'month' column")

    if 'actual_trend' not in df_predictions.columns or 'pred_trend' not in df_predictions.columns:
        raise ValueError("df_predictions must have 'actual_trend' and 'pred_trend' columns")

    # 按月分组计算准确率
    monthly_acc = []
    for month, group in df_predictions.groupby('month'):
        correct = (group['actual_trend'] == group['pred_trend']).sum()
        total = len(group)
        accuracy = (correct / total) * 100 if total > 0 else np.nan
        monthly_acc.append({
            'month': month,
            'trend_accuracy': accuracy
        })

    return pd.DataFrame(monthly_acc)


def add_trend_accuracy_to_monthly(df_monthly, df_predictions):
    """
    为monthly_mape DataFrame添加趋势准确率列

    Args:
        df_monthly: 月度统计DataFrame，必须包含month列
        df_predictions: 预测结果DataFrame，必须包含month, actual_trend, pred_trend列

    Returns:
        DataFrame: 添加了trend_accuracy列的DataFrame
    """
    df_monthly = df_monthly.copy()

    # 计算趋势准确率
    trend_acc_df = calculate_trend_accuracy(df_predictions)

    # 合并到monthly DataFrame
    df_monthly = df_monthly.merge(
        trend_acc_df[['month', 'trend_accuracy']],
        on='month',
        how='left'
    )

    return df_monthly
