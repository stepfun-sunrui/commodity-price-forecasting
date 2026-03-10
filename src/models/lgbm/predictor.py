"""
LightGBM 预测器

实现基于LightGBM的价格预测
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path
from typing import Dict, Any, Union, Tuple

from src.core.base_predictor import BasePredictor
from src.features.targets import get_feature_builder


class LGBMPredictor(BasePredictor):
    """
    LightGBM预测器

    支持两种模式：
    1. predict模式：预测未来N天（在线服务使用）
    2. validate模式：滚动验证（训练时使用）
    """

    def __init__(self):
        """初始化LGBM预测器"""
        super().__init__()
        self.feature_builder_func = None  # 特征构建函数
        self.feature_columns = None  # 保存训练时使用的特征列

    def fit(self, X, y, **kwargs):
        """
        训练模型

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            特征矩阵
        y : pd.Series or np.ndarray
            目标变量
        **kwargs : dict
            LightGBM参数
        """
        # 保存特征列名
        if isinstance(X, pd.DataFrame):
            self.feature_columns = X.columns.tolist()

        # 创建并训练模型
        self.model = lgb.LGBMRegressor(**kwargs)
        self.model.fit(X, y)

        print(f"模型训练完成，使用 {len(self.feature_columns)} 个特征")

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
        if self.model is None:
            model_path = config.get_model_path('lgbm')
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            self.load_model(model_path)

        # 获取特征构建函数
        if self.feature_builder_func is None:
            self.feature_builder_func = get_feature_builder(config.feature_builder)

        # 构建特征
        print(f"构建特征 (使用 {config.feature_builder})...")
        features_df = self.feature_builder_func(data, price_col='price')

        # 根据模式执行预测
        if mode == 'predict':
            return self._predict_future(features_df, config.n_predictions)
        elif mode == 'validate':
            return self._rolling_validation(features_df, data, config)
        else:
            raise ValueError(f"未知的预测模式: {mode}")

    def _predict_future(self, features: pd.DataFrame, n_days: int) -> pd.DataFrame:
        """
        预测未来N天（递归预测）

        Parameters:
        -----------
        features : pd.DataFrame
            特征数据
        n_days : int
            预测天数

        Returns:
        --------
        pd.DataFrame : 预测结果
        """
        print(f"开始预测未来 {n_days} 天...")

        predictions = []
        dates = []
        current_features = features.copy()

        # 获取最后一个日期
        if isinstance(current_features.index, pd.DatetimeIndex):
            last_date = current_features.index[-1]
        else:
            last_date = pd.Timestamp.now()

        # 递归预测
        for i in range(n_days):
            # 获取最后一行特征
            X = current_features.iloc[[-1]][self.feature_columns]

            # 预测
            pred = self.model.predict(X)[0]
            predictions.append(pred)

            # 更新日期
            next_date = last_date + pd.Timedelta(days=i+1)
            dates.append(next_date)

            # 更新特征（简化版：将预测值添加到数据中）
            # 实际应用中可能需要更复杂的特征更新逻辑
            new_row = current_features.iloc[-1].copy()
            new_row['price'] = pred  # 假设价格列名为'price'

            # 将新行添加到特征数据中
            new_row_df = pd.DataFrame([new_row], index=[next_date])
            current_features = pd.concat([current_features, new_row_df])

            # 重新计算特征（可选，取决于特征是否依赖历史数据）
            # 这里简化处理，实际可能需要重新计算滚动特征等

        # 构建结果DataFrame
        result = pd.DataFrame({
            'date': dates,
            'predicted_price': predictions
        })

        print(f"预测完成，生成 {len(predictions)} 个预测点")

        return result

    def _rolling_validation(self, features: pd.DataFrame, data: pd.DataFrame, config) -> Dict[str, Any]:
        """
        滚动验证

        Parameters:
        -----------
        features : pd.DataFrame
            特征数据
        data : pd.DataFrame
            原始数据
        config : TargetConfig
            配置对象

        Returns:
        --------
        dict : 验证结果
        """
        print("开始滚动验证...")

        # 滚动验证参数
        window_size = 252  # 一年交易日
        forecast_horizon = 30  # 预测30天

        results = []
        n_windows = len(data) - window_size - forecast_horizon

        for i in range(0, n_windows, forecast_horizon):
            # 训练窗口
            train_end = window_size + i
            test_end = train_end + forecast_horizon

            if test_end > len(data):
                break

            # 获取训练和测试数据
            train_features = features.iloc[:train_end]
            test_data = data.iloc[train_end:test_end]

            # 训练模型
            X_train = train_features[self.feature_columns]
            y_train = data.iloc[:train_end]['price']  # 假设价格列名为'price'

            temp_model = lgb.LGBMRegressor(**config.model_params)
            temp_model.fit(X_train, y_train)

            # 预测
            predictions = []
            for j in range(forecast_horizon):
                if train_end + j >= len(features):
                    break
                X_test = features.iloc[[train_end + j]][self.feature_columns]
                pred = temp_model.predict(X_test)[0]
                predictions.append(pred)

            # 计算评估指标
            actual = test_data['price'].values[:len(predictions)]
            predictions = np.array(predictions)

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

        # 汇总结果
        summary = {
            'n_windows': len(results),
            'avg_mae': np.mean([r['mae'] for r in results]),
            'avg_rmse': np.mean([r['rmse'] for r in results]),
            'avg_mape': np.mean([r['mape'] for r in results]),
            'details': results
        }

        print(f"滚动验证完成: 平均MAE={summary['avg_mae']:.2f}, 平均RMSE={summary['avg_rmse']:.2f}, 平均MAPE={summary['avg_mape']:.2f}%")

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
            'model': self.model,
            'feature_columns': self.feature_columns
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"模型已保存到: {path}")

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

        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']

        print(f"模型已从 {path} 加载，特征数: {len(self.feature_columns)}")
