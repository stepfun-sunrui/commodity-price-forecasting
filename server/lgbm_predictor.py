"""
LGBM 服务端预测器 - 通用入口

所有标的共用此文件，特征构建通过 src/features/targets/{target} 动态加载
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.targets import get_feature_builder


# ─────────────────────────────────────────────
# 时间特征工具函数
# ─────────────────────────────────────────────

def calendar_features(index: pd.DatetimeIndex, prefix: str = "") -> pd.DataFrame:
    iso = index.isocalendar()
    out = pd.DataFrame(index=index)
    out[f"{prefix}month"] = index.month.astype(int)
    out[f"{prefix}weekofyear"] = iso.week.astype(int)
    out[f"{prefix}dayofweek"] = index.dayofweek.astype(int)
    out[f"{prefix}is_weekend"] = (index.dayofweek >= 5).astype(int)
    out[f"{prefix}is_month_start"] = index.is_month_start.astype(int)
    out[f"{prefix}is_month_end"] = index.is_month_end.astype(int)
    return out


def calendar_features_for_future(origin_index: pd.DatetimeIndex, offset_days: int, prefix: str) -> pd.DataFrame:
    future = origin_index + pd.Timedelta(days=offset_days)
    iso = future.isocalendar()
    out = pd.DataFrame(index=origin_index)
    out[f"{prefix}month"] = np.asarray(future.month, dtype=int)
    out[f"{prefix}weekofyear"] = np.asarray(iso.week, dtype=int)
    out[f"{prefix}dayofweek"] = np.asarray(future.dayofweek, dtype=int)
    out[f"{prefix}is_weekend"] = np.asarray((future.dayofweek >= 5).astype(int), dtype=int)
    out[f"{prefix}is_month_start"] = np.asarray(future.is_month_start.astype(int), dtype=int)
    out[f"{prefix}is_month_end"] = np.asarray(future.is_month_end.astype(int), dtype=int)
    return out


def is_trading_day_flags(index: pd.DatetimeIndex, trading_dates: pd.DatetimeIndex) -> pd.Series:
    trading_set = set(trading_dates.to_pydatetime())
    return pd.Series([1 if d.to_pydatetime() in trading_set else 0 for d in index], index=index, dtype=int)


def days_since_last_trade(index: pd.DatetimeIndex, trading_dates: pd.DatetimeIndex) -> pd.Series:
    trading_set = set(trading_dates.to_pydatetime())
    last_trade = None
    out: list[int] = []
    for ts in index:
        if ts.to_pydatetime() in trading_set:
            last_trade = ts
            out.append(0)
        else:
            out.append(0 if last_trade is None else int((ts - last_trade).days))
    return pd.Series(out, index=index, dtype=int)


def add_basic_time_features(
    features_daily: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    horizon_days: int,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(features_daily.index)
    cal_t = calendar_features(idx, prefix="t_")
    cal_y = calendar_features_for_future(idx, offset_days=horizon_days, prefix=f"tp{horizon_days}_")
    flags = pd.DataFrame(index=idx)
    flags["t_is_trading_day"] = is_trading_day_flags(idx, trading_dates)
    flags["t_days_since_last_trade"] = days_since_last_trade(idx, trading_dates)
    out = pd.concat([features_daily, cal_t, cal_y, flags], axis=1)

    time_cols = [
        c for c in out.columns
        if c.startswith("t_") or c.startswith(f"tp{horizon_days}_") or c == "t_is_trading_day"
    ]
    for c in time_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")
    out["t_days_since_last_trade"] = pd.to_numeric(out["t_days_since_last_trade"], errors="coerce")
    return out


# ─────────────────────────────────────────────
# 训练矩阵构建
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class RollingConfig:
    target: str
    horizon_days: int
    years: int
    min_train_samples: int


def build_training_matrix(
    features_daily: pd.DataFrame,
    real_price: pd.Series,
    trading_dates: pd.DatetimeIndex,
    horizon_days: int,
    train_start: pd.Timestamp,
    t_feature_cutoff: pd.Timestamp,
    daily_price: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    t_idx = features_daily.index
    y_dates = t_idx + pd.Timedelta(days=horizon_days)

    if daily_price is not None:
        train_mask = (
            (t_idx >= train_start)
            & (t_idx <= (t_feature_cutoff - pd.Timedelta(days=horizon_days)))
        )
        X = features_daily.loc[train_mask]
        y = daily_price.reindex(y_dates[train_mask])
    else:
        trading_set = set(trading_dates.to_pydatetime())
        is_trading = np.array([d.to_pydatetime() in trading_set for d in y_dates])
        train_mask = (
            (t_idx >= train_start)
            & (t_idx <= (t_feature_cutoff - pd.Timedelta(days=horizon_days)))
            & is_trading
        )
        X = features_daily.loc[train_mask]
        y = real_price.reindex(y_dates[train_mask])

    tmp = X.join(y.rename("y"), how="inner").dropna(subset=["y"])
    return tmp.drop(columns=["y"]), tmp["y"]


def mape_percent(y_true: float, y_pred: float) -> float:
    if not np.isfinite(y_true) or y_true == 0 or not np.isfinite(y_pred):
        return float("nan")
    return float(abs((y_true - y_pred) / y_true) * 100.0)


# ─────────────────────────────────────────────
# 单点预测
# ─────────────────────────────────────────────

def train_predict_single_point(
    features_daily: pd.DataFrame,
    real_price: pd.Series,
    trading_dates: pd.DatetimeIndex,
    cfg: RollingConfig,
    target_date: pd.Timestamp,
    selected_features: list[str],
    daily_price: pd.Series | None = None,
) -> dict[str, object] | None:
    t_feature = target_date - pd.Timedelta(days=cfg.horizon_days)

    if t_feature < features_daily.index.min():
        return None

    train_end = t_feature
    train_start = train_end - pd.DateOffset(years=cfg.years)

    X_train, y_train = build_training_matrix(
        features_daily=features_daily[selected_features],
        real_price=real_price,
        trading_dates=trading_dates,
        horizon_days=cfg.horizon_days,
        train_start=train_start,
        t_feature_cutoff=train_end,
        daily_price=daily_price,
    )

    if len(X_train) < cfg.min_train_samples:
        return {
            "target_date": target_date,
            "origin_date": t_feature,
            "actual": float(real_price.get(target_date, float("nan"))),
            "pred": float("nan"),
            "mape": float("nan"),
            "n_train": len(X_train),
            "status": "INSUFFICIENT_DATA",
        }

    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                               num_leaves=31, min_child_samples=20, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    x_pred_row = features_daily[selected_features].loc[[t_feature]] if t_feature in features_daily.index else None
    if x_pred_row is None or x_pred_row.isnull().all(axis=None):
        return None

    pred = float(model.predict(x_pred_row)[0])
    actual = float(real_price.get(target_date, float("nan")))
    mape = mape_percent(actual, pred)

    return {
        "target_date": target_date,
        "origin_date": t_feature,
        "actual": actual,
        "pred": pred,
        "mape": mape,
        "n_train": len(X_train),
        "status": "OK",
    }


# ─────────────────────────────────────────────
# 主预测函数（服务端调用入口）
# ─────────────────────────────────────────────

def predict_future(
    target_code: str,
    price_series: pd.Series,
    horizon_days: int,
    years: int,
    n_predictions: int,
    min_train_samples: int,
) -> list[dict]:
    """
    预测未来N个点

    Parameters:
    -----------
    target_code : str
        标的代码，用于加载对应的特征构建函数
    price_series : pd.Series
        历史价格序列（DatetimeIndex）
    horizon_days : int
        预测时域（T+horizon_days）
    years : int
        训练数据年限
    n_predictions : int
        预测点数
    min_train_samples : int
        最小训练样本数

    Returns:
    --------
    list[dict] : 每个元素包含 'date' 和 'value'
    """
    print(f"\n使用通用LGBM预测器...")
    print(f"标的: {target_code}, horizon={horizon_days}, years={years}, n={n_predictions}")

    # 1. 构建日历日序列（ffill）
    daily_index = pd.date_range(price_series.index.min(), price_series.index.max(), freq='D')
    daily_price = price_series.reindex(daily_index).ffill()

    # 限制年限
    end = daily_price.index.max()
    start = end - pd.DateOffset(years=years + 1)
    daily_price = daily_price.loc[start:end]

    trading_dates = pd.DatetimeIndex(price_series.index)
    trading_dates = trading_dates[(trading_dates >= daily_price.index.min()) &
                                  (trading_dates <= daily_price.index.max())]
    real_price = price_series.loc[trading_dates]

    # 2. 构建特征（使用对应标的的特征构建函数）
    print(f"构建特征（使用 {target_code} 特征模块）...")
    feature_builder = get_feature_builder(target_code)
    df_daily = pd.DataFrame({'price': daily_price})
    features_daily = feature_builder(df_daily, price_col='price')

    # 3. 添加时间特征
    features_daily = add_basic_time_features(features_daily, trading_dates, horizon_days)

    # 4. 确定特征列
    model_features = [c for c in features_daily.columns if c not in ['price', 'date']]

    cfg = RollingConfig(
        target=target_code,
        horizon_days=horizon_days,
        years=years,
        min_train_samples=min_train_samples,
    )

    # 5. 生成未来预测日期
    last_date = price_series.index.max()
    future_dates = []
    d = last_date + pd.Timedelta(days=1)
    while len(future_dates) < n_predictions:
        future_dates.append(d)
        d += pd.Timedelta(days=1)

    # 6. 逐点预测
    print(f"开始逐点预测，共 {n_predictions} 个点...")
    predictions = []

    for i, target_date in enumerate(future_dates, 1):
        t_feature = target_date - pd.Timedelta(days=horizon_days)

        if t_feature < features_daily.index.min():
            continue

        train_end = t_feature
        train_start = train_end - pd.DateOffset(years=years)

        X_train, y_train = build_training_matrix(
            features_daily=features_daily[model_features],
            real_price=real_price,
            trading_dates=trading_dates,
            horizon_days=horizon_days,
            train_start=train_start,
            t_feature_cutoff=train_end,
            daily_price=daily_price,
        )

        if len(X_train) < min_train_samples:
            print(f"  [{i}/{n_predictions}] {target_date.date()} 训练样本不足 ({len(X_train)}), 跳过")
            continue

        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                                   num_leaves=31, min_child_samples=20, subsample=0.8,
                                   colsample_bytree=0.8, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        if t_feature not in features_daily.index:
            # 用最近的特征行
            available = features_daily.index[features_daily.index <= t_feature]
            if len(available) == 0:
                continue
            t_feature_use = available[-1]
        else:
            t_feature_use = t_feature

        x_pred = features_daily[model_features].loc[[t_feature_use]]
        pred = float(model.predict(x_pred)[0])

        print(f"  [{i}/{n_predictions}] 预测 {target_date.date()}: {pred:.2f}")

        predictions.append({
            'date': target_date,
            'value': pred,
        })

    print(f"LGBM预测完成: {len(predictions)} 个预测点")
    return predictions
