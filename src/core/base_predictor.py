"""
预测器基类

定义所有预测器的统一接口
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Union
from pathlib import Path


class BasePredictor(ABC):
    """
    预测器抽象基类

    所有预测器（LGBM, SARIMA, Ensemble）都需要继承这个类
    并实现其抽象方法
    """

    def __init__(self):
        self.model = None

    @abstractmethod
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
            模型参数
        """
        pass

    @abstractmethod
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
            预测模式：
            - 'predict': 预测未来N天（在线服务使用）
            - 'validate': 滚动验证（训练时使用）

        Returns:
        --------
        预测结果（格式取决于mode）
        """
        pass

    @abstractmethod
    def save_model(self, path: Union[str, Path]):
        """
        保存模型到文件

        Parameters:
        -----------
        path : str or Path
            模型保存路径
        """
        pass

    @abstractmethod
    def load_model(self, path: Union[str, Path]):
        """
        从文件加载模型

        Parameters:
        -----------
        path : str or Path
            模型文件路径
        """
        pass

    def _predict_future(self, features: pd.DataFrame, n_days: int) -> pd.DataFrame:
        """
        预测未来N天（递归预测）

        子类可以重写这个方法以实现特定的预测逻辑

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
        raise NotImplementedError("Subclass must implement _predict_future method")

    def _rolling_validation(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
        """
        滚动验证

        子类可以重写这个方法以实现特定的验证逻辑

        Parameters:
        -----------
        features : pd.DataFrame
            特征数据
        data : pd.DataFrame
            原始数据

        Returns:
        --------
        dict : 验证结果（包含各种评估指标）
        """
        raise NotImplementedError("Subclass must implement _rolling_validation method")
