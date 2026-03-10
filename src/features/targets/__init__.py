"""
目标特征构建模块
为每个预测标的提供独立的特征构建函数

目录结构：
targets/
├── ID00102866/
│   └── ID00102866_features.py
├── ID00103568/
│   └── ID00103568_features.py
├── ID00103617/
│   ├── ID00103617_features.py
│   ├── ID00103617_features_full.py
│   └── ID00103617_jump_features.py
├── ID01020441/
│   └── ID01020441_features.py
├── ID01560197/
│   └── ID01560197_features.py
└── RE00035675/
    └── RE00035675_features.py
"""

# 导入各标的的特征构建函数
from .ID00102866.ID00102866_features import build_features as build_ID00102866_features
from .ID00103568.ID00103568_features import build_features as build_ID00103568_features
from .ID00103617.ID00103617_features import build_features as build_ID00103617_features
from .ID01020441.ID01020441_features import build_features as build_ID01020441_features
from .ID01560197.ID01560197_features import build_features as build_ID01560197_features
from .RE00035675.RE00035675_features import build_features as build_RE00035675_features

# 特征构建函数映射
FEATURE_BUILDERS = {
    'ID00102866': build_ID00102866_features,
    'ID00103568': build_ID00103568_features,
    'ID00103617': build_ID00103617_features,
    'ID01020441': build_ID01020441_features,
    'ID01560197': build_ID01560197_features,
    'RE00035675': build_RE00035675_features,
}


def get_feature_builder(target_code: str):
    """
    获取指定标的的特征构建函数
    
    Args:
        target_code: 标的代码
    
    Returns:
        特征构建函数
    
    Raises:
        ValueError: 如果标的代码不存在
    """
    if target_code not in FEATURE_BUILDERS:
        raise ValueError(f'Unknown target code: {target_code}. Available: {list(FEATURE_BUILDERS.keys())}')
    return FEATURE_BUILDERS[target_code]


__all__ = [
    'build_ID00102866_features',
    'build_ID00103568_features',
    'build_ID00103617_features',
    'build_ID01020441_features',
    'build_ID01560197_features',
    'build_RE00035675_features',
    'FEATURE_BUILDERS',
    'get_feature_builder',
]
