# 特征构建模块

## 目录结构

```
src/features/targets/
├── __init__.py                    # 模块入口
├── ID00102866/                    # 冶金焦
│   └── ID00102866_features.py     # 109个特征
├── ID00103568/                    # 材料指标1
│   └── ID00103568_features.py     # 88个特征
├── ID00103617/                    # 材料指标2
│   └── ID00103617_features.py     # 138个特征
├── ID01020441/                    # 材料指标3
│   └── ID01020441_features.py     # 119个特征
├── ID01560197/                    # 材料指标4
│   └── ID01560197_features.py     # 43个特征
└── RE00035675/                    # 焦煤
    └── RE00035675_features.py     # 93个特征
```

## 使用方法

```python
from src.features.targets import get_feature_builder

# 获取特定标的的特征构建函数
builder = get_feature_builder('ID00102866')

# 构建特征
features_df = builder(df, price_col='price')
```

## 特征统计

| 标的代码 | 名称 | 特征数 | 关键特征 |
|---------|------|--------|---------|
| ID00102866 | 冶金焦 | 109个 | 月度和季节性特征 |
| ID00103617 | 材料指标2 | 138个 | 完整跳跃预测特征 |
| ID01020441 | 材料指标3 | 119个 | 相对位置、加速度等 |
| RE00035675 | 焦煤 | 93个 | 技术指标（EMA、BB、RSI） |
| ID00103568 | 材料指标1 | 88个 | 相对位置、动量等 |
| ID01560197 | 材料指标4 | 43个 | 基础特征 |
