# 商品价格预测系统

基于 LightGBM / SARIMA 的商品价格滚动预测系统，支持离线验证和在线服务两种模式。

## 项目结构

```
├── configs/
│   └── targets/              # 各标的配置文件（YAML）
│       ├── ID00102866.yaml   # 冶金焦
│       ├── ID00103568.yaml
│       ├── ID00103617.yaml
│       ├── ID01020441.yaml
│       ├── ID01560197.yaml
│       └── RE00035675.yaml   # 焦煤
│
├── data/
│   └── need_predict_data/    # 各标的历史价格 CSV（{target}_cleaned.csv）
│
├── src/
│   ├── core/
│   │   ├── config_manager.py # 读取 configs/targets/*.yaml
│   │   ├── data_processor.py # 数据解析
│   │   └── base_predictor.py
│   ├── features/
│   │   └── targets/          # 各标的特征构建函数
│   │       ├── ID00102866/ID00102866_features.py
│   │       ├── ID00103568/ID00103568_features.py
│   │       ├── ID00103617/ID00103617_features.py
│   │       ├── ID01020441/ID01020441_features.py
│   │       ├── ID01560197/ID01560197_features.py
│   │       └── RE00035675/RE00035675_features.py
│   └── models/
│       ├── lgbm/predictor.py   # 供 training 使用的 LGBM 滚动预测器
│       └── sarima/predictor.py # 供 training 使用的 SARIMA 滚动预测器
│
├── server/
│   ├── app.py                # FastAPI 服务入口
│   ├── predict.py            # 核心预测逻辑（PricePrediction）
│   ├── lgbm_predictor.py     # 通用 LGBM 预测器（服务端）
│   ├── sarima_wrapper.py     # SARIMA 预测包装器（读取 configs）
│   ├── data_processor.py     # 输入 JSON 解析
│   ├── test_prediction.py    # 不启动服务直接测试
│   ├── data/                 # 保存接收到的输入数据
│   └── output/               # 保存预测结果 JSON
│
└── training/
    └── scripts/
        └── rolling_forecast.py  # 离线滚动验证入口
```

## 预测标的

| 代码 | 名称 |
|------|------|
| ID00102866 | 冶金焦 |
| ID00103568 | 材料指标1 |
| ID00103617 | 材料指标2 |
| ID01020441 | 材料指标3 |
| ID01560197 | 材料指标4 |
| RE00035675 | 焦煤 |

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn lightgbm statsmodels pandas numpy scikit-learn pyyaml requests
```

### 2. 离线滚动验证

编辑 `training/scripts/rolling_forecast.py` 中的 `PREDICTION_TARGETS` 列表，选择要验证的标的，然后运行：

```bash
python training/scripts/rolling_forecast.py
```

支持命令行参数：

```bash
# 指定标的和模型
python training/scripts/rolling_forecast.py --target ID00102866 --model lgbm

# 指定验证窗口
python training/scripts/rolling_forecast.py --target RE00035675 --model sarima --windows 12
```

### 3. 服务端预测

**不启动服务，直接测试：**

```bash
cd server
python test_prediction.py
```

读取 `server/input_*.json`，结果保存到 `server/output/prediction_{target}_{timestamp}.json`。

**启动 FastAPI 服务：**

```bash
cd server
uvicorn app:app --host 0.0.0.0 --port 8000
```

服务接收 POST 请求，对每个标的完成预测后自动回调。

## 配置说明

每个标的的配置文件位于 `configs/targets/{target}.yaml`，控制：

- `active_model`：使用的模型（`lgbm` / `sarima` / `ensemble`）
- `lgbm.data.years`：LGBM 训练使用的历史年限
- `sarima.best_params`：SARIMA 的 order 和 seasonal_order
- `prediction.n_predictions`：预测天数

## 输出格式

```json
{
  "PRE_MATERIAL_PRICE_OUTPUT": [
    {
      "PRICE_DATE": "2026-02-01T00:00:00Z",
      "PRICE_VALUE": 823.50,
      "PRICE_VALUE_UB": 850.00,
      "PRICE_VALUE_LB": 780.00
    }
  ],
  "PRE_CHARACTER_FACTOR_OUTPUT": [...],
  "PRE_INFLUENCE_FACTOR_OUTPUT": [...]
}
```

## 架构说明

- `src/` 提供特征构建和模型实现，被 `training/` 和 `server/` 共同复用
- `server/lgbm_predictor.py` 是服务端通用 LGBM 入口，通过 `get_feature_builder(target_code)` 动态加载各标的特征
- `server/sarima_wrapper.py` 从 `configs/targets/*.yaml` 读取 SARIMA 超参数
- 模型参数、历史数据年限均通过 `configs/` 统一管理，无需修改代码
