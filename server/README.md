# 价格预测服务

## 目录结构

```
server/
├── app.py                    # FastAPI服务入口
├── predict.py                # 预测核心逻辑（可直接运行测试）
├── data_processor.py         # 数据处理器
├── sarima_wrapper.py         # SARIMA预测包装器
├── test_prediction.py        # 测试脚本（推荐使用）
├── data/                     # 输入数据保存目录
├── output/                   # 预测结果输出目录
└── input_20260130_155428.json  # 测试数据样例
```

## 使用方式

### 方式1: 使用test_prediction.py（推荐）

**优点：** 从配置文件读取参数，支持LGBM和SARIMA，自动处理多个标的

```bash
cd server

# 使用默认测试数据
python test_prediction.py

# 使用指定的JSON文件
python test_prediction.py path/to/input.json
```

**特点：**
- ✅ 自动从 `configs/targets/*.yaml` 读取配置
- ✅ 支持LGBM和SARIMA模型
- ✅ 自动选择正确的数据年限
- ✅ 输出包含50天预测 + 市场相关性分析

### 方式2: 直接运行predict.py

**注意：** 使用硬编码参数，仅用于快速测试

```bash
cd server
python predict.py
```

**限制：**
- ⚠️ 硬编码参数（years=5, prediction_mode="lgbm"）
- ⚠️ 不会从配置文件读取
- ⚠️ 需要手动修改代码来改变参数

### 方式3: 启动FastAPI服务

```bash
cd server
python app.py
```

服务地址: `http://localhost:6001`

**API接口：**
- `POST /predict` - 预测接口
- `GET /health` - 健康检查
- `GET /config` - 配置信息

## 测试数据格式

输入JSON格式：
```json
{
  "TAR_INDEX_CODE": ["ID00102866", "RE00035675"],
  "ALG_ID": "AIYC2601300004",
  "PRE_FEATURE_INFO": [
    {
      "INDEX_CODE": "ID00102866",
      "DATA_VALUE": 1500.0,
      "DATA_DATE": "2025-10-28 00:00:00.0",
      "FREQUENCY": "DAT",
      "INDEX_NAME": "冶金焦价格"
    }
  ]
}
```

## 输出格式

预测结果保存在 `server/output/prediction_{target_code}_{timestamp}.json`：

```json
{
  "response_metadata": {
    "request_id": "PRED_AIYC2601300004",
    "response_time": "2026-02-13T08:55:07Z",
    "status": "success"
  },
  "PRE_MATERIAL_PRICE_OUTPUT": [
    {
      "ALG_ID": "AIYC2601300004",
      "FREQUENCY": "日度",
      "MTRL_NO": "IRON_ID00102866",
      "INDEX_CODE": "ID00102866",
      "PRICE_DATE": "2026-01-28T00:00:00Z",
      "PRICE_VALUE": 1638.24,
      "PRICE_VALUE_UB": 1720.15,
      "PRICE_VALUE_LB": 1556.33
    }
  ],
  "PRE_CHARACTER_FACTOR_OUTPUT": [
    {
      "INDEX_CODE": "FEATURE_001",
      "INDEX_NAME": "特征1",
      "CORRELATION": 0.85
    }
  ],
  "PRE_INFLUENCE_FACTOR_OUTPUT": [
    {
      "INDEX_CODE": "FEATURE_001",
      "INDEX_NAME": "特征1",
      "CORRELATION": 0.85,
      "CHANGE_RATE": 0.05
    }
  ]
}
```

## 配置系统

所有预测配置从 `configs/targets/{target_code}.yaml` 读取：

```yaml
target:
  code: "ID00102866"
  name: "冶金焦"

active_model: "lgbm"  # lgbm/sarima

lgbm:
  data:
    years: 2  # 使用最近2年数据

sarima:
  data:
    years: 1
  best_params:
    order: [1, 1, 1]
    seasonal_order: [1, 0, 1, 30]

prediction:
  horizon_days: 30
  n_predictions: 50
```

## 代码复用

1. **SARIMA配置** - 从 `configs/targets/*.yaml` 读取超参数
2. **LGBM特征** - 从 `src/features/targets/{target_code}_features.py` 复用
3. **数据处理** - 使用统一的 `data_processor.py`
4. **相关性分析** - 自动计算市场相关性

## 支持的标的

| 标的代码 | 名称 | 模型类型 | 数据年限 |
|---------|------|---------|---------|
| RE00035675 | 焦煤 | LGBM | 5年 |
| ID00102866 | 冶金焦 | LGBM | 2年 |
| ID00103568 | 材料指标1 | LGBM | 5年 |
| ID00103617 | 材料指标2 | SARIMA | 3年 |
| ID01020441 | 材料指标3 | SARIMA | 2年 |
| ID01560197 | 材料指标4 | SARIMA | 1年 |

## 常见问题

### Q: 如何不启动服务直接测试？

**A: 使用 `test_prediction.py`（推荐）**
```bash
python test_prediction.py
```

或者直接运行 `predict.py`（使用硬编码参数）：
```bash
python predict.py
```

### Q: test_prediction.py 和 predict.py 有什么区别？

| 特性 | test_prediction.py | predict.py |
|-----|-------------------|-----------|
| 配置读取 | ✅ 从YAML读取 | ❌ 硬编码 |
| 模型选择 | ✅ 自动选择 | ❌ 固定lgbm |
| 数据年限 | ✅ 按配置 | ❌ 固定5年 |
| 推荐使用 | ✅ 是 | ⚠️ 仅快速测试 |

### Q: 如何修改预测参数？

编辑 `configs/targets/{target_code}.yaml` 文件，无需重启服务。

### Q: 输出文件在哪里？

- 测试输出: `server/output/prediction_{target_code}_{timestamp}.json`
- 服务输入保存: `server/data/input_{timestamp}.json`

## 实现检查

详见 `IMPLEMENTATION_CHECK.md` - 包含与old版本的对比和所有功能点的验证。

