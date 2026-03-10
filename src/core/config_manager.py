"""
配置管理器

负责加载和管理标的配置
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """
    配置管理器

    负责加载和保存标的配置文件
    """

    @staticmethod
    def load_target(target_code: str) -> 'TargetConfig':
        """
        加载标的配置

        Parameters:
        -----------
        target_code : str
            标的代码（如 "ID00102866"）

        Returns:
        --------
        TargetConfig : 标的配置对象
        """
        # 查找配置文件路径
        config_path = Path(__file__).parent.parent.parent / "configs" / "targets" / f"{target_code}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 加载YAML配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return TargetConfig(config_dict, target_code)

    @staticmethod
    def load_model_config(model_type: str) -> Dict[str, Any]:
        """
        加载模型配置

        Parameters:
        -----------
        model_type : str
            模型类型（如 "lgbm", "sarima"）

        Returns:
        --------
        dict : 模型配置字典
        """
        config_path = Path(__file__).parent.parent.parent / "configs" / "models" / f"{model_type}.yaml"

        if not config_path.exists():
            return {}

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)


class TargetConfig:
    """
    标的配置类

    封装单个标的的所有配置信息
    """

    def __init__(self, config_dict: Dict[str, Any], target_code: str):
        """
        初始化标的配置

        Parameters:
        -----------
        config_dict : dict
            配置字典
        target_code : str
            标的代码
        """
        self.target_code = target_code
        self.config_dict = config_dict  # 保存原始配置

        # 基本信息
        target_info = config_dict.get('target', {})
        self.code = target_info.get('code', target_code)
        self.name = target_info.get('name', '')
        self.description = target_info.get('description', '')

        # 数据配置（通用）
        self.data = config_dict.get('data', {})
        self.min_train_samples = self.data.get('min_train_samples', 300)

        # 特征构建器
        self.feature_builder = config_dict.get('feature_builder', target_code)

        # 当前激活的模型类型
        self.model_type = config_dict.get('active_model', 'lgbm')

        # LGBM配置
        self.lgbm_config = config_dict.get('lgbm', {})

        # SARIMA配置
        self.sarima_config = config_dict.get('sarima', {})

        # 根据当前激活的模型类型设置参数
        if self.model_type == 'lgbm':
            self.years = self.lgbm_config.get('data', {}).get('years', 5)
            self.model_params = self.lgbm_config.get('best_params', {})
        elif self.model_type == 'sarima':
            self.years = self.sarima_config.get('data', {}).get('years', 5)
            self.model_params = self.sarima_config.get('best_params', {})
        else:
            self.years = 5
            self.model_params = {}

        # 预测配置
        self.prediction = config_dict.get('prediction', {})
        self.horizon_days = self.prediction.get('horizon_days', 30)
        self.n_predictions = self.prediction.get('n_predictions', 50)

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        获取指定模型的配置

        Parameters:
        -----------
        model_type : str
            模型类型 ('lgbm' 或 'sarima')

        Returns:
        --------
        dict : 模型配置
        """
        if model_type == 'lgbm':
            return self.lgbm_config
        elif model_type == 'sarima':
            return self.sarima_config
        else:
            return {}

    def save(self):
        """
        保存配置到文件

        将当前配置保存回YAML文件
        """
        config_dict = {
            'target': {
                'code': self.code,
                'name': self.name,
                'description': self.description
            },
            'data': {
                'years': self.years,
                'min_train_samples': self.min_train_samples
            },
            'features': self.features,
            'model': {
                'type': self.model_type,
                'best_params': self.model_params
            },
            'prediction': {
                'horizon_days': self.horizon_days,
                'n_predictions': self.n_predictions
            }
        }

        config_path = Path(__file__).parent.parent.parent / "configs" / "targets" / f"{self.target_code}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)

    def get_model_path(self, model_type: Optional[str] = None) -> Path:
        """
        获取模型文件路径

        Parameters:
        -----------
        model_type : str, optional
            模型类型，如果不指定则使用配置中的类型

        Returns:
        --------
        Path : 模型文件路径
        """
        if model_type is None:
            model_type = self.model_type

        models_dir = Path(__file__).parent.parent.parent / "training" / "outputs" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        return models_dir / f"{self.target_code}_{model_type}.pkl"

    def __repr__(self):
        return f"TargetConfig(code={self.code}, name={self.name}, model_type={self.model_type})"
