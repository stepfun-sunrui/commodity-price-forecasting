"""
滚动预测脚本 - 使用新架构

支持LGBM和SARIMA模型的滚动验证
"""
import warnings
import os

# 必须在导入其他库之前设置
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager
from src.models.lgbm.predictor import LGBMPredictor
from src.models.sarima.predictor import SARIMAPredictor
from src.features.targets import get_feature_builder


# ========================================
# 配置：需要预测的标的列表
# ========================================
# 可以在这里添加或删除标的代码
PREDICTION_TARGETS = [
    'ID00102866',  # 冶金焦
    'ID00103568',  # 材料指标1
    'RE00035675',  # 焦煤
    'ID00103617',  # 材料指标2
    'ID01020441',  # 材料指标3
    'ID01560197',  # 材料指标4
]

# 标的信息（用于显示）
TARGET_INFO = {
    'ID00102866': '冶金焦',
    'ID00103568': '材料指标1',
    'RE00035675': '焦煤',
    'ID00103617': '材料指标2',
    'ID01020441': '材料指标3',
    'ID01560197': '材料指标4',
}
# ========================================


def load_data(target_code: str, config) -> pd.DataFrame:
    """加载数据"""
    # 尝试多个可能的路径
    possible_paths = [
        project_root / "data" / "need_predict_data" / f"{target_code}_cleaned.csv",
        project_root / "need_predict_data" / f"{target_code}_cleaned.csv",
        project_root / "data" / "processed" / f"{target_code}_cleaned.csv",
        project_root / "data" / "processed" / f"{target_code}.csv",
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(f"数据文件不存在，尝试了以下路径:\n" + "\n".join(str(p) for p in possible_paths))

    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    df = df.sort_index()

    # 根据配置筛选数据年限
    if config.years > 0:
        cutoff_date = df.index[-1] - pd.DateOffset(years=config.years)
        df = df[df.index >= cutoff_date]
        print(f"使用最近 {config.years} 年数据: {len(df)} 条记录")

    return df


def rolling_forecast_lgbm(target_code: str, val_months: int = 6):
    """
    LGBM月底滚动预测

    在每个月月底，用历史数据训练模型，预测下个月一整个月的点位

    Parameters:
    -----------
    target_code : str
        标的代码
    val_months : int
        验证集月数
    """
    print(f"\n{'='*60}")
    print(f"LGBM月底滚动预测: {target_code}")
    print(f"{'='*60}\n")

    # 1. 加载配置
    print("[1/5] 加载配置...")
    config = ConfigManager.load_target(target_code)
    print(f"  - 标的名称: {config.name}")
    print(f"  - 预测天数: {config.n_predictions}")

    # 2. 加载数据
    print("\n[2/5] 加载数据...")
    data = load_data(target_code, config)
    print(f"  - 数据形状: {data.shape}")
    print(f"  - 日期范围: {data.index[0]} 到 {data.index[-1]}")

    # 3. 构建特征
    print("\n[3/5] 构建特征...")
    feature_builder = get_feature_builder(config.feature_builder)
    features_df = feature_builder(data, price_col='price')
    print(f"  - 特征数量: {len(features_df.columns)}")

    # 删除NaN
    features_df = features_df.dropna()
    print(f"  - 删除NaN后样本数: {len(features_df)}")

    # 4. 月底滚动预测
    print("\n[4/5] 开始月底滚动预测...")

    # 找到所有月底日期
    all_month_ends = features_df.resample('ME').last().index

    # 计算验证集起始月份
    val_start_date = features_df.index[-1] - pd.DateOffset(months=val_months)
    val_month_ends = [me for me in all_month_ends if me >= val_start_date]

    # 确保有足够的训练数据
    if len(val_month_ends) == 0:
        raise ValueError(f"验证集月数 {val_months} 太大，没有足够的数据")

    print(f"  - 验证集月份数: {len(val_month_ends)}")
    print(f"  - 第一个验证月: {val_month_ends[0].strftime('%Y-%m')}")
    print(f"  - 最后一个验证月: {val_month_ends[-1].strftime('%Y-%m')}")

    predictions = []
    actuals = []
    dates = []

    # 在每个月月底进行预测
    for month_idx, month_end in enumerate(val_month_ends[:-1], 1):  # 最后一个月没有下个月数据
        print(f"\n  [{month_idx}/{len(val_month_ends)-1}] 预测 {month_end.strftime('%Y-%m')} 的下个月...")

        # 训练集：到当前月底的所有数据
        train_df = features_df[features_df.index <= month_end]

        if len(train_df) < config.min_train_samples:
            print(f"    跳过：训练样本不足 ({len(train_df)} < {config.min_train_samples})")
            continue

        # 准备训练数据
        y_train = train_df['price'].shift(-1).iloc[:-1]
        X_train = train_df.drop(columns=['price'], errors='ignore').iloc[:-1]

        # 训练模型
        predictor = LGBMPredictor()
        predictor.fit(X_train, y_train, **config.model_params)

        # 预测下个月的所有点位
        next_month_start = month_end + pd.DateOffset(days=1)
        next_month_end = (next_month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

        # 获取下个月的实际日期
        next_month_data = features_df[(features_df.index > month_end) &
                                      (features_df.index <= next_month_end)]

        if len(next_month_data) == 0:
            print(f"    跳过：下个月没有数据")
            continue

        print(f"    训练样本: {len(train_df)}, 预测天数: {len(next_month_data)}")

        # 使用月底最后一天的特征预测整个下个月
        # 获取月底最后一天的特征
        month_end_features = train_df.iloc[-1:].drop(columns=['price'], errors='ignore')

        # 对下个月的每一天，都使用月底特征进行预测
        for pred_date in next_month_data.index:
            # 使用月底特征预测
            y_pred = predictor.model.predict(month_end_features)[0]

            # 实际值
            y_actual = features_df.loc[pred_date, 'price']

            predictions.append(y_pred)
            actuals.append(y_actual)
            dates.append(pred_date)

    # 5. 计算指标
    print("\n[5/5] 计算评估指标...")

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    print(f"\n滚动预测结果:")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAPE: {mape:.4f}%")

    # 保存每日结果
    results_df = pd.DataFrame({
        'date': dates,
        'actual': actuals,
        'predicted': predictions,
        'error': actuals - predictions,
        'abs_error': np.abs(actuals - predictions),
        'pct_error': np.abs((actuals - predictions) / actuals) * 100
    })

    # 保存到 outputs/daily/ 目录
    output_daily_dir = project_root / "training" / "outputs" / "daily"
    output_daily_dir.mkdir(parents=True, exist_ok=True)

    # 文件名格式：{标的代码}_daily_lgbm.csv
    daily_output_path = output_daily_dir / f"{target_code}_daily_lgbm.csv"
    results_df.to_csv(daily_output_path, index=False, encoding='utf-8-sig')
    print(f"\n每日结果已保存到: {daily_output_path}")

    # 计算月度统计（参考旧代码格式）
    results_df['month'] = pd.to_datetime(results_df['date']).dt.to_period('M').astype(str)

    # 获取原始数据用于计算前一个月的均价
    original_data = features_df[['price']].copy()
    original_data['month'] = original_data.index.to_period('M').astype(str)

    monthly_stats = []
    for month in sorted(results_df['month'].unique()):
        month_data = results_df[results_df['month'] == month]

        # 每日MAPE的平均值
        daily_mape_mean = month_data['pct_error'].mean()

        # 月均价MAPE
        month_actual_mean = month_data['actual'].mean()
        month_pred_mean = month_data['predicted'].mean()

        # 计算月均价MAPE
        if month_actual_mean > 0:
            month_avg_mape = abs((month_actual_mean - month_pred_mean) / month_actual_mean) * 100
        else:
            month_avg_mape = 0

        # 计算上个月的实际均价（从原始数据中获取）
        month_date = pd.Period(month)
        prev_month = str(month_date - 1)

        # 从原始数据中查找上个月的均价
        if prev_month in original_data['month'].values:
            prev_month_data = original_data[original_data['month'] == prev_month]
            prev_month_actual_mean = prev_month_data['price'].mean()
        else:
            prev_month_actual_mean = np.nan

        # 计算趋势（使用±20规则）
        if not np.isnan(prev_month_actual_mean) and prev_month_actual_mean > 0:
            actual_diff = month_actual_mean - prev_month_actual_mean
            pred_diff = month_pred_mean - prev_month_actual_mean

            # 使用±20规则判断趋势
            if actual_diff > 20:
                actual_trend_vs_prev = "上涨"
            elif actual_diff < -20:
                actual_trend_vs_prev = "下跌"
            else:
                actual_trend_vs_prev = "不变"

            if pred_diff > 20:
                pred_trend_vs_prev = "上涨"
            elif pred_diff < -20:
                pred_trend_vs_prev = "下跌"
            else:
                pred_trend_vs_prev = "不变"

            trend_correct = 1.0 if actual_trend_vs_prev == pred_trend_vs_prev else 0.0
        else:
            actual_trend_vs_prev = "N/A"
            pred_trend_vs_prev = "N/A"
            trend_correct = 0.0

        monthly_stats.append({
            'month': month,
            'daily_mape_mean': daily_mape_mean,
            'daily_mape_corrected_mean': daily_mape_mean,  # 与daily_mape_mean相同
            'month_avg_mape': month_avg_mape,
            'n_points': len(month_data),
            'month_actual_mean': month_actual_mean,
            'month_pred_mean': month_pred_mean,
            'prev_month_actual_mean': prev_month_actual_mean,
            'actual_trend_vs_prev': actual_trend_vs_prev,
            'pred_trend_vs_prev': pred_trend_vs_prev,
            'trend_correct_vs_prev': trend_correct,
        })

    monthly_df = pd.DataFrame(monthly_stats)

    # 添加Overall行
    overall_daily_mape = results_df['pct_error'].mean()
    overall_actual_mean = results_df['actual'].mean()
    overall_pred_mean = results_df['predicted'].mean()
    overall_month_avg_mape = monthly_df['month_avg_mape'].mean()
    overall_count = len(results_df)
    overall_trend_correct_rate = monthly_df['trend_correct_vs_prev'].mean() * 100

    overall_row = pd.DataFrame({
        'month': ['Overall'],
        'daily_mape_mean': [overall_daily_mape],
        'daily_mape_corrected_mean': [overall_daily_mape],
        'month_avg_mape': [overall_month_avg_mape],
        'n_points': [overall_count],
        'month_actual_mean': [overall_actual_mean],
        'month_pred_mean': [overall_pred_mean],
        'prev_month_actual_mean': [np.nan],
        'actual_trend_vs_prev': ['N/A'],
        'pred_trend_vs_prev': ['N/A'],
        'trend_correct_vs_prev': [overall_trend_correct_rate],
    })

    monthly_df = pd.concat([monthly_df, overall_row], ignore_index=True)

    # 保存到 outputs/monthly/ 目录
    output_monthly_dir = project_root / "training" / "outputs" / "monthly"
    output_monthly_dir.mkdir(parents=True, exist_ok=True)

    # 文件名格式：{标的代码}_monthly_lgbm.csv
    monthly_output_path = output_monthly_dir / f"{target_code}_monthly_lgbm.csv"
    monthly_df.to_csv(monthly_output_path, index=False, encoding='utf-8-sig')
    print(f"月度统计已保存到: {monthly_output_path}")

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': predictions,
        'actuals': actuals,
        'dates': dates
    }


def rolling_forecast_sarima(target_code: str, val_months: int = 6):
    """
    SARIMA月底滚动预测

    在每个月月底，用历史数据训练模型，预测下个月一整个月的点位

    Parameters:
    -----------
    target_code : str
        标的代码
    val_months : int
        验证集月数
    """
    print(f"\n{'='*60}")
    print(f"SARIMA月底滚动预测: {target_code}")
    print(f"{'='*60}\n")

    # 1. 加载配置
    print("[1/4] 加载配置...")
    config = ConfigManager.load_target(target_code)

    # 获取SARIMA参数（直接从sarima配置中读取）
    sarima_config = config.sarima_config
    if not sarima_config or 'best_params' not in sarima_config:
        raise ValueError(f"{target_code} 没有配置SARIMA参数，请检查配置文件")

    order = sarima_config['best_params'].get('order')
    seasonal_order = sarima_config['best_params'].get('seasonal_order')

    if not order or not seasonal_order:
        raise ValueError(f"{target_code} 的SARIMA参数不完整，请检查配置文件")

    print(f"  - 标的名称: {config.name}")
    print(f"  - SARIMA参数: order={order}, seasonal_order={seasonal_order}")

    # 2. 加载数据
    print("\n[2/4] 加载数据...")
    data = load_data(target_code, config)
    print(f"  - 数据形状: {data.shape}")
    print(f"  - 日期范围: {data.index[0]} 到 {data.index[-1]}")

    # 3. 月底滚动预测
    print("\n[3/4] 开始月底滚动预测...")

    # 找到所有月底日期
    all_month_ends = data.resample('ME').last().index

    # 计算验证集起始月份
    val_start_date = data.index[-1] - pd.DateOffset(months=val_months)
    val_month_ends = [me for me in all_month_ends if me >= val_start_date]

    # 确保有足够的训练数据
    if len(val_month_ends) == 0:
        raise ValueError(f"验证集月数 {val_months} 太大，没有足够的数据")

    print(f"  - 验证集月份数: {len(val_month_ends)}")
    print(f"  - 第一个验证月: {val_month_ends[0].strftime('%Y-%m')}")
    print(f"  - 最后一个验证月: {val_month_ends[-1].strftime('%Y-%m')}")

    predictions = []
    actuals = []
    dates = []

    predictor = SARIMAPredictor()

    # 在每个月月底进行预测
    for month_idx, month_end in enumerate(val_month_ends[:-1], 1):  # 最后一个月没有下个月数据
        print(f"\n  [{month_idx}/{len(val_month_ends)-1}] 预测 {month_end.strftime('%Y-%m')} 的下个月...")

        # 训练集：到当前月底的所有数据
        train_data = data[data.index <= month_end]

        if len(train_data) < config.min_train_samples:
            print(f"    跳过：训练样本不足 ({len(train_data)} < {config.min_train_samples})")
            continue

        # 训练模型（抑制警告）
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor.fit(
                train_data['price'],
                order=order,
                seasonal_order=seasonal_order
            )

        # 预测下个月的所有点位
        next_month_start = month_end + pd.DateOffset(days=1)
        next_month_end = (next_month_start + pd.DateOffset(months=1)) - pd.DateOffset(days=1)

        # 获取下个月的实际交易日期
        next_month_data = data[(data.index > month_end) &
                               (data.index <= next_month_end)]

        if len(next_month_data) == 0:
            print(f"    跳过：下个月没有数据")
            continue

        print(f"    训练样本: {len(train_data)}, 预测天数: {len(next_month_data)}")

        # 对下个月的每个交易日进行预测（与旧代码逻辑一致）
        for target_date in next_month_data.index:
            # 计算从月底到目标日期的天数
            days_ahead = (target_date - month_end).days

            if days_ahead <= 0:
                continue

            # 预测days_ahead步，取最后一个值（抑制警告）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = predictor.fitted_model.forecast(steps=days_ahead)
            pred_value = forecast.iloc[-1] if hasattr(forecast, 'iloc') else forecast[-1]

            # 实际值
            actual_value = next_month_data.loc[target_date, 'price']

            predictions.append(pred_value)
            actuals.append(actual_value)
            dates.append(target_date)

    # 4. 计算指标
    print("\n[4/4] 计算评估指标...")

    # 转换为numpy数组并过滤NaN
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 过滤掉NaN值
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions_valid = predictions[valid_mask]
    actuals_valid = actuals[valid_mask]

    if len(predictions_valid) == 0:
        print("警告：没有有效的预测结果")
        mae = rmse = mape = 0
    else:
        mae = np.mean(np.abs(actuals_valid - predictions_valid))
        rmse = np.sqrt(np.mean((actuals_valid - predictions_valid) ** 2))
        mape = np.mean(np.abs((actuals_valid - predictions_valid) / actuals_valid)) * 100

    print(f"\n滚动预测结果:")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - MAPE: {mape:.4f}%")

    # 保存每日结果
    results_df = pd.DataFrame({
        'date': dates,
        'actual': actuals,
        'predicted': predictions,
        'error': actuals - predictions,
        'abs_error': np.abs(actuals - predictions),
        'pct_error': np.abs((actuals - predictions) / actuals) * 100
    })

    # 保存到 outputs/daily/ 目录
    output_daily_dir = project_root / "training" / "outputs" / "daily"
    output_daily_dir.mkdir(parents=True, exist_ok=True)

    # 文件名格式：{标的代码}_daily_sarima.csv
    daily_output_path = output_daily_dir / f"{target_code}_daily_sarima.csv"
    results_df.to_csv(daily_output_path, index=False, encoding='utf-8-sig')
    print(f"\n每日结果已保存到: {daily_output_path}")

    # 计算月度统计（过滤掉NaN值，参考旧代码格式）
    results_df_valid = results_df.dropna()

    if len(results_df_valid) > 0:
        results_df_valid['month'] = pd.to_datetime(results_df_valid['date']).dt.to_period('M').astype(str)

        # 获取原始数据用于计算前一个月的均价
        original_data = data[['price']].copy()
        original_data['month'] = original_data.index.to_period('M').astype(str)

        monthly_stats = []
        for month in sorted(results_df_valid['month'].unique()):
            month_data = results_df_valid[results_df_valid['month'] == month]

            # 每日MAPE的平均值
            daily_mape_mean = month_data['pct_error'].mean()

            # 月均价MAPE
            month_actual_mean = month_data['actual'].mean()
            month_pred_mean = month_data['predicted'].mean()

            # 计算月均价MAPE
            if month_actual_mean > 0:
                month_avg_mape = abs((month_actual_mean - month_pred_mean) / month_actual_mean) * 100
            else:
                month_avg_mape = 0

            # 计算上个月的实际均价（从原始数据中获取）
            month_date = pd.Period(month)
            prev_month = str(month_date - 1)

            # 从原始数据中查找上个月的均价
            if prev_month in original_data['month'].values:
                prev_month_data = original_data[original_data['month'] == prev_month]
                prev_month_actual_mean = prev_month_data['price'].mean()
            else:
                prev_month_actual_mean = np.nan

            # 计算趋势（使用±15规则）
            if not np.isnan(prev_month_actual_mean) and prev_month_actual_mean > 0:
                actual_diff = month_actual_mean - prev_month_actual_mean
                pred_diff = month_pred_mean - prev_month_actual_mean

                # 使用±15规则判断趋势
                if actual_diff > 15:
                    actual_trend_vs_prev = "上涨"
                elif actual_diff < -15:
                    actual_trend_vs_prev = "下跌"
                else:
                    actual_trend_vs_prev = "不变"

                if pred_diff > 15:
                    pred_trend_vs_prev = "上涨"
                elif pred_diff < -15:
                    pred_trend_vs_prev = "下跌"
                else:
                    pred_trend_vs_prev = "不变"

                trend_correct = 1.0 if actual_trend_vs_prev == pred_trend_vs_prev else 0.0
            else:
                actual_trend_vs_prev = "N/A"
                pred_trend_vs_prev = "N/A"
                trend_correct = 0.0

            monthly_stats.append({
                'month': month,
                'daily_mape_mean': daily_mape_mean,
                'daily_mape_corrected_mean': daily_mape_mean,  # 与daily_mape_mean相同
                'month_avg_mape': month_avg_mape,
                'n_points': len(month_data),
                'month_actual_mean': month_actual_mean,
                'month_pred_mean': month_pred_mean,
                'prev_month_actual_mean': prev_month_actual_mean,
                'actual_trend_vs_prev': actual_trend_vs_prev,
                'pred_trend_vs_prev': pred_trend_vs_prev,
                'trend_correct_vs_prev': trend_correct,
            })

        monthly_df = pd.DataFrame(monthly_stats)

        # 添加Overall行
        overall_daily_mape = results_df_valid['pct_error'].mean()
        overall_actual_mean = results_df_valid['actual'].mean()
        overall_pred_mean = results_df_valid['predicted'].mean()
        overall_month_avg_mape = monthly_df['month_avg_mape'].mean()
        overall_count = len(results_df_valid)
        overall_trend_correct_rate = monthly_df['trend_correct_vs_prev'].mean() * 100

        overall_row = pd.DataFrame({
            'month': ['Overall'],
            'daily_mape_mean': [overall_daily_mape],
            'daily_mape_corrected_mean': [overall_daily_mape],
            'month_avg_mape': [overall_month_avg_mape],
            'n_points': [overall_count],
            'month_actual_mean': [overall_actual_mean],
            'month_pred_mean': [overall_pred_mean],
            'prev_month_actual_mean': [np.nan],
            'actual_trend_vs_prev': ['N/A'],
            'pred_trend_vs_prev': ['N/A'],
            'trend_correct_vs_prev': [overall_trend_correct_rate],
        })

        monthly_df = pd.concat([monthly_df, overall_row], ignore_index=True)

        # 保存到 outputs/monthly/ 目录
        output_monthly_dir = project_root / "training" / "outputs" / "monthly"
        output_monthly_dir.mkdir(parents=True, exist_ok=True)

        # 文件名格式：{标的代码}_monthly_sarima.csv
        monthly_output_path = output_monthly_dir / f"{target_code}_monthly_sarima.csv"
        monthly_df.to_csv(monthly_output_path, index=False, encoding='utf-8-sig')
        print(f"月度统计已保存到: {monthly_output_path}")

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': predictions,
        'actuals': actuals,
        'dates': dates
    }


def main():
    parser = argparse.ArgumentParser(
        description='滚动预测脚本 - 默认测试所有配置的标的',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试所有标的（默认）
  python rolling_forecast.py

  # 测试所有标的，指定验证集月数
  python rolling_forecast.py --val-months 3

  # 测试单个标的
  python rolling_forecast.py --target ID00102866

  # 测试单个标的，指定模型
  python rolling_forecast.py --target ID00102866 --model lgbm --val-months 3
        """
    )
    parser.add_argument('--target', type=str, default=None,
                        help='标的代码（不指定则测试所有配置的标的）')
    parser.add_argument('--model', type=str, choices=['lgbm', 'sarima', 'auto'], default='sarima',
                        help='模型类型 (auto: 使用配置文件中的active_model)')
    parser.add_argument('--val-months', type=int, default=6, help='验证集月数')

    args = parser.parse_args()

    # 确定要测试的标的列表
    if args.target:
        # 测试单个标的
        targets_to_test = [args.target]
    else:
        # 测试所有配置的标的
        targets_to_test = PREDICTION_TARGETS

    print(f"\n{'='*60}")
    print(f"滚动预测批量测试")
    print(f"{'='*60}")
    print(f"标的数量: {len(targets_to_test)}")
    print(f"验证集月数: {args.val_months}")
    print(f"{'='*60}\n")

    # 存储所有结果
    all_results = []

    # 逐个测试标的
    for idx, target_code in enumerate(targets_to_test, 1):
        target_name = TARGET_INFO.get(target_code, target_code)

        print(f"\n{'#'*60}")
        print(f"[{idx}/{len(targets_to_test)}] 测试标的: {target_code} ({target_name})")
        print(f"{'#'*60}")

        try:
            # 加载配置确定模型类型
            config = ConfigManager.load_target(target_code)

            if args.model == 'auto':
                model_type = config.model_type
            else:
                model_type = args.model

            print(f"\n使用模型: {model_type}")

            # 执行滚动预测
            if model_type == 'lgbm':
                results = rolling_forecast_lgbm(target_code, args.val_months)
            elif model_type == 'sarima':
                results = rolling_forecast_sarima(target_code, args.val_months)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

            # 保存结果
            all_results.append({
                'target_code': target_code,
                'target_name': target_name,
                'model_type': model_type,
                'mae': results['mae'],
                'rmse': results['rmse'],
                'mape': results['mape'],
                'status': 'success'
            })

            print(f"\n[OK] {target_code} 测试完成")

        except Exception as e:
            print(f"\n[FAIL] {target_code} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()

            all_results.append({
                'target_code': target_code,
                'target_name': target_name,
                'model_type': 'unknown',
                'mae': 0,
                'rmse': 0,
                'mape': 0,
                'status': f'failed: {str(e)}'
            })

    # 打印汇总结果
    print(f"\n{'='*60}")
    print(f"测试汇总")
    print(f"{'='*60}\n")

    if all_results:
        # 创建汇总表格
        summary_df = pd.DataFrame(all_results)

        # 格式化显示
        print(f"{'标的代码':<15} {'名称':<12} {'模型':<8} {'MAPE':<10} {'状态':<10}")
        print(f"{'-'*60}")

        for _, row in summary_df.iterrows():
            mape_str = f"{row['mape']:.2f}%" if row['status'] == 'success' else 'N/A'
            status_str = '[OK]' if row['status'] == 'success' else '[FAIL]'
            print(f"{row['target_code']:<15} {row['target_name']:<12} {row['model_type']:<8} {mape_str:<10} {status_str:<10}")

        # 统计
        success_count = len(summary_df[summary_df['status'] == 'success'])
        total_count = len(summary_df)

        print(f"\n{'-'*60}")
        print(f"成功: {success_count}/{total_count}")

        if success_count > 0:
            avg_mape = summary_df[summary_df['status'] == 'success']['mape'].mean()
            print(f"平均MAPE: {avg_mape:.2f}%")

        print(f"\n结果保存在:")
        print(f"  - training/outputs/daily/{{标的代码}}_daily.csv")
        print(f"  - training/outputs/monthly/{{标的代码}}_monthly.csv")

    print(f"\n{'='*60}")
    print(f"所有测试完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
