"""
SARIMA 预测包装器

使用新架构的配置系统和SARIMA预测器
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict
import warnings

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False
    print("Warning: statsmodels not available")


def predict_sarima(
    df_train: pd.DataFrame,
    target_code: str,
    n_predictions: int
) -> List[Dict]:
    """
    使用SARIMA进行预测

    Parameters:
    -----------
    df_train : pd.DataFrame
        训练数据，包含 'date' 和 'price' 列
    target_code : str
        目标标的代码
    n_predictions : int
        预测天数

    Returns:
    --------
    list : 预测结果列表，每个元素包含 'date' 和 'value'
    """
    if not SARIMAX_AVAILABLE:
        raise ImportError("statsmodels is required for SARIMA prediction")

    print(f"\n使用新架构SARIMA预测器...")

    # 加载配置
    config = ConfigManager.load_target(target_code)
    sarima_config = config.sarima_config

    # 获取SARIMA参数
    order = tuple(sarima_config['best_params']['order'])
    seasonal_order = tuple(sarima_config['best_params']['seasonal_order'])

    print(f"SARIMA参数: order={order}, seasonal_order={seasonal_order}")

    # 准备数据
    if 'date' in df_train.columns:
        df_train = df_train.set_index('date')

    price_series = df_train['price']

    # 训练模型
    print(f"训练SARIMA模型，数据点数: {len(price_series)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            price_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted_model = model.fit(disp=False)

    print("SARIMA模型训练完成")

    # 预测
    print(f"预测未来 {n_predictions} 天...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = fitted_model.forecast(steps=n_predictions)

    # 生成预测日期
    last_date = price_series.index[-1]
    pred_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_predictions,
        freq='D'
    )

    # 构建预测结果
    predictions = []
    for date, value in zip(pred_dates, forecast):
        predictions.append({
            'date': pd.Timestamp(date),
            'value': float(value)
        })

    print(f"SARIMA预测完成: {len(predictions)} 个预测点")

    return predictions
