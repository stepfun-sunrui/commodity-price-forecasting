"""
价格预测模块
继承 PredictionDataProcessor，根据 target_index_code 调用对应的预测脚本
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from data_processor import PredictionDataProcessor


class PricePrediction(PredictionDataProcessor):
    """
    价格预测类

    继承自 PredictionDataProcessor，添加预测功能
    """

    def __init__(self, data_source: Union[str, Path, dict]):
        """
        初始化预测器

        Parameters:
        -----------
        data_source : str, Path, or dict
            输入数据源
        """
        super().__init__(data_source)
        self.sarima_module_path = None  # SARIMA 模块路径
        self._load_prediction_module()

    def _load_prediction_module(self):
        """根据 target_index_code 加载对应的预测模块"""
        target_code = self.get_target_index_code()

        server_dir = Path(__file__).parent

        # 查找 SARIMA 模块（旧架构）
        sarima_path = server_dir / target_code / "sarima_analysis_t30" / "sarima_forecast.py"
        if sarima_path.exists():
            self.sarima_module_path = sarima_path
            print(f"找到 SARIMA 模块: {sarima_path}")

    def _import_module_from_path(self, module_path: Path, module_name: str):
        """动态导入模块"""
        # 添加必要的路径到 sys.path
        repo_root = Path(__file__).parent.parent
        ml_dir = repo_root / "ml"
        module_dir = module_path.parent  # 模块所在目录

        # 添加顺序很重要：模块目录优先级最高
        paths_to_add = [str(module_dir), str(ml_dir), str(repo_root)]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def prepare_data_for_prediction(self) -> pd.DataFrame:
        """
        准备预测所需的数据格式

        Returns:
        --------
        pd.DataFrame : 包含 date 和 price 列的 DataFrame

        Raises:
        -------
        ValueError : 如果数据中存在重复日期
        """
        target_df = self.get_target_dataframe()

        # 转换为预测脚本需要的格式
        df = pd.DataFrame({
            'date': target_df['DATA_DATE'],
            'price': target_df['DATA_VALUE']
        })

        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        # 检查是否存在重复日期
        duplicate_dates = df[df.duplicated(subset=['date'], keep=False)]
        if len(duplicate_dates) > 0:
            unique_dup_dates = duplicate_dates['date'].unique()
            error_msg = f"数据错误：检测到 {len(unique_dup_dates)} 个重复日期！\n"
            error_msg += f"重复日期示例（前10个）:\n"
            for date in unique_dup_dates[:10]:
                count = len(duplicate_dates[duplicate_dates['date'] == date])
                error_msg += f"  - {date}: 出现 {count} 次\n"
            if len(unique_dup_dates) > 10:
                error_msg += f"  ... 还有 {len(unique_dup_dates) - 10} 个重复日期\n"
            error_msg += "\n请检查并清理输入数据中的重复日期后重试。"
            raise ValueError(error_msg)

        return df

    def predict(
        self,
        horizon_days: int = 30,
        years: int = None,
        min_train_samples: int = 300,
        n_predictions: int = 30,
        prediction_mode: str = "lgbm"
    ) -> List[Dict]:
        """
        进行价格预测

        Parameters:
        -----------
        horizon_days : int, default=30
            预测时域（T+horizon_days）
        years : int, optional
            使用最近 N 年的数据训练，如果 None 或数据不足则使用全部数据
        min_train_samples : int, default=300
            最小训练样本数
        n_predictions : int, default=30
            预测未来 N 个交易日
        prediction_mode : str, default="lgbm"
            预测模式：
            - "lgbm": 只使用 LightGBM（默认）
            - "sarima": 只使用 SARIMA
            - "ensemble": 使用集成（LightGBM + SARIMA）

        Returns:
        --------
        list of dict : 预测结果列表，每个元素包含:
            - PRICE_DATE: 预测日期
            - PRICE_VALUE: 预测价格
            - PRICE_VALUE_UB: 预测上界
            - PRICE_VALUE_LB: 预测下界
        """
        # 验证预测模式
        valid_modes = ["lgbm", "sarima", "ensemble"]
        if prediction_mode not in valid_modes:
            raise ValueError(f"无效的预测模式: {prediction_mode}，必须是 {valid_modes} 之一")

        print(f"\n开始预测 {self.get_target_index_code()} 的未来价格...")
        print(f"预测模式: {prediction_mode.upper()}")

        # 准备数据
        df = self.prepare_data_for_prediction()
        print(f"数据点数: {len(df)}")
        print(f"日期范围: {df['date'].min()} 至 {df['date'].max()}")

        # 检查数据量
        if len(df) < min_train_samples:
            print(f"警告: 数据量 ({len(df)}) 少于最小训练样本数 ({min_train_samples})")
            print(f"将使用所有可用数据进行训练")

        # 如果指定了 years 参数，截取最近 N 年数据
        if years is not None:
            end_date = df['date'].max()
            start_date = end_date - pd.DateOffset(years=years)
            df_train = df[df['date'] >= start_date].copy()

            if len(df_train) < min_train_samples:
                print(f"最近 {years} 年数据不足 {min_train_samples} 个样本")
                print(f"使用全部 {len(df)} 个数据点进行训练")
                df_train = df.copy()
            else:
                print(f"使用最近 {years} 年的数据: {len(df_train)} 个样本")
        else:
            df_train = df.copy()
            print(f"使用全部数据进行训练: {len(df_train)} 个样本")

        # 调用预测方法
        if prediction_mode == "sarima":
            # 只使用 SARIMA 预测
            if not self.sarima_module_path:
                # 使用新架构的SARIMA预测器
                print("使用新架构SARIMA预测器...")
                predictions = self._predict_with_sarima_wrapper(
                    df_train, n_predictions
                )
            else:
                predictions = self._predict_with_sarima_only(
                    df_train, n_predictions
                )
        elif prediction_mode in ["lgbm", "ensemble"]:
            # 使用通用 LGBM 预测器
            predictions = self._predict_with_lgbm_predictor(
                df_train, horizon_days, n_predictions, min_train_samples, prediction_mode
            )
        else:
            # 使用通用预测方法
            predictions = self._predict_with_generic_method(
                df_train, horizon_days, n_predictions
            )

        print(f"预测完成，生成 {len(predictions)} 个预测点")
        return predictions

    def _predict_with_lgbm_predictor(
        self,
        df_train: pd.DataFrame,
        horizon_days: int,
        n_predictions: int,
        min_train_samples: int = 300,
        prediction_mode: str = "lgbm"
    ) -> List[Dict]:
        """使用通用 LGBM 预测器进行预测"""
        from lgbm_predictor import predict_future

        target_code = self.get_target_index_code()

        # 准备价格序列
        price_series = df_train.set_index('date')['price']

        # 调用通用 LGBM 预测器（years=100 表示使用 df_train 中的全部数据）
        pred_values = predict_future(
            target_code=target_code,
            price_series=price_series,
            horizon_days=horizon_days,
            years=100,
            n_predictions=n_predictions,
            min_train_samples=min_train_samples,
        )

        if not pred_values:
            raise ValueError("LGBM 预测未返回任何结果")

        # ensemble 模式：叠加 SARIMA
        if prediction_mode == "ensemble":
            try:
                sarima_preds = self._predict_with_sarima_wrapper(df_train, n_predictions)
                sarima_dict = {
                    pd.Timestamp(p['PRICE_DATE'].rstrip('Z')): p['PRICE_VALUE']
                    for p in sarima_preds
                }
                lgbm_w, sarima_w = 0.6, 0.4
                for p in pred_values:
                    ts = pd.Timestamp(p['date'])
                    if ts in sarima_dict:
                        p['value'] = p['value'] * lgbm_w + sarima_dict[ts] * sarima_w
                print(f"集成预测完成（LGBM {lgbm_w} + SARIMA {sarima_w}）")
            except Exception as e:
                print(f"警告: SARIMA 预测失败，仅使用 LGBM: {e}")

        # 计算上下界（基于前30个预测点）
        first_30 = pred_values[:min(30, len(pred_values))]
        values = [p['value'] for p in first_30]
        price_ub = max(values)
        price_lb = min(values)

        print(f"价格上界 (UB): {price_ub:.2f}, 下界 (LB): {price_lb:.2f}")

        predictions = []
        for p in pred_values:
            predictions.append({
                'PRICE_DATE': pd.Timestamp(p['date']).isoformat() + 'Z',
                'PRICE_VALUE': round(p['value'], 2),
                'PRICE_VALUE_UB': round(price_ub, 2),
                'PRICE_VALUE_LB': round(price_lb, 2),
            })

        return predictions

    def _predict_with_sarima_only(
        self,
        df_train: pd.DataFrame,
        n_predictions: int
    ) -> List[Dict]:
        """只使用 SARIMA 进行预测"""
        print("\n只使用 SARIMA 预测模式...")

        # 导入 SARIMA 模块
        sarima_module = self._import_module_from_path(
            self.sarima_module_path,
            f"sarima_forecast_{self.get_target_index_code()}"
        )

        # 调用 SARIMA 预测
        sarima_predictions = sarima_module.predict_sarima(
            df_train=df_train,
            target_code=self.get_target_index_code(),
            n_predictions=n_predictions
        )

        if not sarima_predictions:
            raise ValueError("SARIMA 预测失败，未返回任何预测值")

        print(f"\nSARIMA 预测成功: {len(sarima_predictions)} 个预测点")

        # 计算上下界
        first_30_days = sarima_predictions[:min(30, len(sarima_predictions))]
        if first_30_days:
            values = [p['value'] for p in first_30_days]
            price_ub = max(values)
            price_lb = min(values)
        else:
            price_ub = 0
            price_lb = 0

        print(f"基于前 {len(first_30_days)} 天 SARIMA 预测计算上下界:")
        print(f"  价格上界 (UB): {price_ub:.2f}")
        print(f"  价格下界 (LB): {price_lb:.2f}")

        # 生成预测结果
        predictions = []
        for pred in sarima_predictions:
            predictions.append({
                'PRICE_DATE': pred['date'].isoformat() + 'Z',
                'PRICE_VALUE': round(pred['value'], 2),
                'PRICE_VALUE_UB': round(price_ub, 2),
                'PRICE_VALUE_LB': round(price_lb, 2),
            })

        return predictions

    def _predict_with_sarima_wrapper(
        self,
        df_train: pd.DataFrame,
        n_predictions: int
    ) -> List[Dict]:
        """使用新架构的SARIMA预测器"""
        from sarima_wrapper import predict_sarima

        # 调用SARIMA包装器
        sarima_predictions = predict_sarima(
            df_train=df_train,
            target_code=self.get_target_index_code(),
            n_predictions=n_predictions
        )

        if not sarima_predictions:
            raise ValueError("SARIMA 预测失败，未返回任何预测值")

        print(f"\nSARIMA 预测成功: {len(sarima_predictions)} 个预测点")

        # 计算上下界
        first_30_days = sarima_predictions[:min(30, len(sarima_predictions))]
        if first_30_days:
            values = [p['value'] for p in first_30_days]
            price_ub = max(values)
            price_lb = min(values)
        else:
            price_ub = 0
            price_lb = 0

        print(f"基于前 {len(first_30_days)} 天 SARIMA 预测计算上下界:")
        print(f"  价格上界 (UB): {price_ub:.2f}")
        print(f"  价格下界 (LB): {price_lb:.2f}")

        # 生成预测结果
        predictions = []
        for pred in sarima_predictions:
            predictions.append({
                'PRICE_DATE': pred['date'].isoformat() + 'Z',
                'PRICE_VALUE': round(pred['value'], 2),
                'PRICE_VALUE_UB': round(price_ub, 2),
                'PRICE_VALUE_LB': round(price_lb, 2),
            })

        return predictions

    def _predict_with_generic_method(
        self,
        df_train: pd.DataFrame,
        horizon_days: int,
        n_predictions: int
    ) -> List[Dict]:
        """使用通用预测方法"""
        print("\n使用通用预测方法...")
        print("警告: 这是一个简化的预测方法，可能精度较低")

        # 简单方法：使用移动平均
        last_price = df_train['price'].iloc[-1]
        last_date = df_train['date'].iloc[-1]

        predictions = []
        for i in range(1, n_predictions + 1):
            pred_date = last_date + timedelta(days=i)
            # 跳过周末
            while pred_date.weekday() >= 5:
                pred_date += timedelta(days=1)

            # 简单预测：使用最后价格加小幅波动
            pred_value = last_price * (1 + np.random.uniform(-0.02, 0.02))
            pred_ub = pred_value * 1.05
            pred_lb = pred_value * 0.95

            predictions.append({
                'PRICE_DATE': pred_date.isoformat() + 'Z',
                'PRICE_VALUE': round(pred_value, 2),
                'PRICE_VALUE_UB': round(pred_ub, 2),
                'PRICE_VALUE_LB': round(pred_lb, 2),
            })

        return predictions

    def predict_and_generate_output(
        self,
        horizon_days: int = 30,
        years: int = None,
        min_train_samples: int = 300,
        n_predictions: int = 30,
        alg_id: str = None,
        run_time: str = None,
        prediction_mode: str = "lgbm",
        lag_days: int = 30
    ) -> Dict:
        """
        执行预测并生成标准输出格式

        Parameters:
        -----------
        horizon_days : int, default=30
            预测时域
        years : int, optional
            使用最近 N 年数据，如果不足则使用全部
        min_train_samples : int, default=300
            最小训练样本数
        n_predictions : int, default=30
            预测未来 N 个交易日
        alg_id : str, optional
            算法 ID
        run_time : str, optional
            运行时间
        prediction_mode : str, default="lgbm"
            预测模式：
            - "lgbm": 只使用 LightGBM（默认）
            - "sarima": 只使用 SARIMA
            - "ensemble": 使用集成（LightGBM + SARIMA）
        lag_days : int, default=30
            前置天数，用于计算特征相关性

        Returns:
        --------
        dict : 标准输出格式
        """
        # 记录开始时间
        start_time = datetime.now().isoformat() + "Z"

        # 执行预测
        predictions = self.predict(
            horizon_days=horizon_days,
            years=years,
            min_train_samples=min_train_samples,
            n_predictions=n_predictions,
            prediction_mode=prediction_mode
        )

        # 记录结束时间
        end_time = datetime.now().isoformat() + "Z"

        # 生成标准输出
        output = self.generate_output_format(
            predictions=predictions,
            alg_id=alg_id,
            run_time=run_time,
            start_time=start_time,
            end_time=end_time
        )

        # 计算前置相关性
        print(f"\n计算前置 {lag_days} 天的特征相关性...")
        correlations_df = self.compute_lagged_correlations(lag_days=lag_days)
        print(f"计算完成，共 {len(correlations_df)} 个特征")

        # 生成两个新表
        character_factor_output = self.generate_character_factor_output(
            correlations_df=correlations_df,
            alg_id=alg_id,
            run_time=run_time
        )

        influence_factor_output = self.generate_influence_factor_output(
            correlations_df=correlations_df,
            alg_id=alg_id,
            run_time=run_time,
            top_n=3
        )

        # 添加到输出
        output['PRE_CHARACTER_FACTOR_OUTPUT'] = character_factor_output
        output['PRE_INFLUENCE_FACTOR_OUTPUT'] = influence_factor_output

        return output


# 使用示例
if __name__ == "__main__":
    import json

    print("=" * 80)
    print("多标的价格预测测试")
    print("=" * 80)

    # 加载输入数据（使用最新的测试数据）
    json_file = Path(__file__).parent / "input_20260130_155428.json"

    if not json_file.exists():
        print(f"错误: 文件不存在 {json_file}")
        print("请确保测试数据文件存在")
        exit(1)

    # 读取数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取目标标的列表
    tar_index_code = data.get('TAR_INDEX_CODE')
    if isinstance(tar_index_code, str):
        target_codes = [tar_index_code]
    elif isinstance(tar_index_code, list):
        target_codes = tar_index_code
    else:
        print(f"错误: TAR_INDEX_CODE 格式不正确")
        exit(1)

    print(f"\n目标标的列表: {target_codes}")
    print(f"共需预测 {len(target_codes)} 个标的")
    print(f"总记录数: {len(data.get('PRE_FEATURE_INFO', []))}")

    # 物料编号映射
    MTRL_NO_MAPPING = {
        "RE00035675": "COAL_RE00035675",
        "ID00102866": "IRON_ID00102866",
        "ID00103568": "MTRL_ID00103568",
        "ID00103617": "MTRL_ID00103617",
        "ID01020441": "MTRL_ID01020441",
        "ID01560197": "MTRL_ID01560197",
    }

    # 输出目录
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # 顺序处理每个标的
    success_count = 0
    fail_count = 0

    for idx, target_code in enumerate(target_codes, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(target_codes)}] 处理标的: {target_code}")
        print(f"{'='*80}")

        try:
            # 创建数据副本
            data_copy = data.copy()
            data_copy['TAR_INDEX_CODE'] = target_code

            # 过滤 PRE_FEATURE_INFO，移除其他目标标的的数据
            original_count = len(data_copy['PRE_FEATURE_INFO'])
            filtered_features = [
                record for record in data_copy['PRE_FEATURE_INFO']
                if record.get('INDEX_CODE') == target_code or record.get('INDEX_CODE') not in target_codes
            ]
            data_copy['PRE_FEATURE_INFO'] = filtered_features
            filtered_count = len(filtered_features)

            print(f"数据过滤: {original_count} -> {filtered_count} 条记录")

            # 创建预测器
            predictor = PricePrediction(data_copy)

            # 设置物料编号
            mtrl_no = MTRL_NO_MAPPING.get(target_code, f"MTRL_{target_code}")
            predictor.set_mtrl_no(mtrl_no)
            print(f"物料编号: {mtrl_no}")

            # 执行预测并生成输出
            print(f"开始预测...")
            output = predictor.predict_and_generate_output(
                horizon_days=30,
                years=5,  # 使用最近5年数据，如果不足则使用全部
                n_predictions=50,
                prediction_mode="lgbm",
                alg_id=data.get('ALG_ID', 'TEST_001')
            )

            # 保存输出结果
            timestamp = datetime.now().strftime("%Y%m%d")
            output_file = output_dir / f"prediction_{target_code}_{timestamp}.json"

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f"\n[成功] 预测完成！")
            print(f"  输出文件: {output_file}")
            print(f"  预测点数: {len(output.get('PRE_MATERIAL_PRICE_OUTPUT', []))}")
            print(f"  特征因子数: {len(output.get('PRE_CHARACTER_FACTOR_OUTPUT', []))}")
            print(f"  影响因素记录数: {len(output.get('PRE_INFLUENCE_FACTOR_OUTPUT', []))}")

            success_count += 1

        except Exception as e:
            print(f"\n[失败] 预测失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    # 汇总结果
    print(f"\n{'='*80}")
    print("测试结果汇总")
    print(f"{'='*80}")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"总计: {len(target_codes)} 个")
    print("=" * 80)