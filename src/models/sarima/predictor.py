"""
SARIMA 预测器

实现基于SARIMA的时间序列预测
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Union, Tuple

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False
    print("Warning: statsmodels not available, SARIMA predictor will not work")

from src.core.base_predictor import BasePredictor


class SARIMAPredictor(BasePredictor):
    """
    SARIMA预测器

    支持两种模式：
    1. predict模式：预测未来N天（在线服务使用）
    2. validate模式：滚动验证（训练时使用）
    """

    def __init__(self):
        """初始化SARIMA预测器"""
        super().__init__()
        self.order = None
        self.seasonal_order = None
        self.fitted_model = None

    def fit(self, data, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7), **kwargs):
        """
        训练SARIMA模型

        Parameters:
        -----------
        data : pd.Series or pd.DataFrame
            时间序列数据
        order : tuple
            ARIMA阶数 (p, d, q)
        seasonal_order : tuple
            季节性阶数 (P, D, Q, s)
        **kwargs : dict
            其他SARIMAX参数
        """
        if not SARIMAX_AVAILABLE:
            raise ImportError("statsmodels is required for SARIMA predictor")

        self.order = order
        self.seasonal_order = seasonal_order

        # 如果是DataFrame，提取价格列
        if isinstance(data, pd.DataFrame):
            if 'price' in data.columns:
                data = data['price']
            else:
                data = data.iloc[:, 0]

        # 训练模型
        print(f"训练SARIMA模型: order={order}, seasonal_order={seasonal_order}")
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order, **kwargs)
        self.fitted_model = model.fit(disp=False)

        print(f"SARIMA模型训练完成")

    def predict(self, data: pd.DataFrame, config, mode: str = 'predict'):
        """
        预测方法

        Parameters:
        -----------
        data : pd.DataFrame
            输入数据
        config : TargetConfig
            标的配置对象
        mode : str
            'predict' - 预测未来N天
            'validate' - 滚动验证

        Returns:
        --------
        预测结果
        """
        # 加载模型（如果还没加载）
        if self.fitted_model is None:
            model_path = config.get_model_path('sarima')
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            self.load_model(model_path)

        # 根据模式执行预测
        if mode == 'predict':
            return self._predict_future(data, config.n_predictions)
        elif mode == 'validate':
            return self._rolling_validation(data, config)
        else:
            raise ValueError(f"未知的预测模式: {mode}")

    def _predict_future(self, data: pd.DataFrame, n_days: int) -> pd.DataFrame:
        """
        预测未来N天

        Parameters:
        -----------
        data : pd.DataFrame
            历史数据
        n_days : int
            预测天数

        Returns:
        --------
        pd.DataFrame : 预测结果
        """
        print(f"开始SARIMA预测未来 {n_days} 天...")

        # 提取价格序列
        if 'price' in data.columns:
            price_series = data['price']
        else:
            price_series = data.iloc[:, 0]

        # 使用已训练的模型进行预测
        forecast = self.fitted_model.forecast(steps=n_days)

        # 生成日期
        if isinstance(data.index, pd.DatetimeIndex):
            last_date = data.index[-1]
            dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days, freq='D')
        else:
            dates = range(len(data), len(data) + n_days)

        # 构建结果DataFrame
        result = pd.DataFrame({
            'date': dates,
            'predicted_price': forecast.values
        })

        print(f"SARIMA预测完成，生成 {len(forecast)} 个预测点")

        return result

    def _rolling_validation(self, data: pd.DataFrame, config) -> Dict[str, Any]:
        """
        滚动验证

        Parameters:
        -----------
        data : pd.DataFrame
            原始数据
        config : TargetConfig
            配置对象

        Returns:
        --------
        dict : 验证结果
        """
        print("开始SARIMA滚动验证...")

        # 提取价格序列
        if 'price' in data.columns:
            price_series = data['price']
        else:
            price_series = data.iloc[:, 0]

        # 滚动验证参数
        window_size = 252  # 一年交易日
        forecast_horizon = 30  # 预测30天

        results = []
        n_windows = len(price_series) - window_size - forecast_horizon

        # 获取模型参数
        order = config.model_params.get('order', self.order or (1, 1, 1))
        seasonal_order = config.model_params.get('seasonal_order', self.seasonal_order or (1, 0, 1, 7))

        for i in range(0, n_windows, forecast_horizon):
            train_end = window_size + i
            test_end = train_end + forecast_horizon

            if test_end > len(price_series):
                break

            # 训练数据
            train_data = price_series.iloc[:train_end]

            try:
                # 训练模型
                model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
                fitted = model.fit(disp=False)

                # 预测
                forecast = fitted.forecast(steps=forecast_horizon)

                # 实际值
                actual = price_series.iloc[train_end:test_end].values

                # 计算评估指标
                predictions = forecast.values[:len(actual)]
                mae = np.mean(np.abs(actual - predictions))
                rmse = np.sqrt(np.mean((actual - predictions) ** 2))
                mape = np.mean(np.abs((actual - predictions) / actual)) * 100

                results.append({
                    'window': i // forecast_horizon,
                    'train_end': train_end,
                    'test_end': test_end,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape
                })

                print(f"  窗口 {i // forecast_horizon}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

            except Exception as e:
                print(f"  窗口 {i // forecast_horizon} 训练失败: {str(e)}")
                continue

        # 汇总结果
        if results:
            summary = {
                'n_windows': len(results),
                'avg_mae': np.mean([r['mae'] for r in results]),
                'avg_rmse': np.mean([r['rmse'] for r in results]),
                'avg_mape': np.mean([r['mape'] for r in results]),
                'details': results
            }

            print(f"SARIMA滚动验证完成: 平均MAE={summary['avg_mae']:.2f}, 平均RMSE={summary['avg_rmse']:.2f}, 平均MAPE={summary['avg_mape']:.2f}%")
        else:
            summary = {
                'n_windows': 0,
                'avg_mae': 0,
                'avg_rmse': 0,
                'avg_mape': 0,
                'details': []
            }
            print("SARIMA滚动验证失败：没有成功的验证窗口")

        return summary

    def save_model(self, path: Union[str, Path]):
        """
        保存模型到文件

        Parameters:
        -----------
        path : str or Path
            模型保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'fitted_model': self.fitted_model,
            'order': self.order,
            'seasonal_order': self.seasonal_order
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"SARIMA模型已保存到: {path}")

    def load_model(self, path: Union[str, Path]):
        """
        从文件加载模型

        Parameters:
        -----------
        path : str or Path
            模型文件路径
        """
        path = Path(path)

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.fitted_model = model_data['fitted_model']
        self.order = model_data['order']
        self.seasonal_order = model_data['seasonal_order']

        print(f"SARIMA模型已从 {path} 加载")
