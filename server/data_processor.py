"""
数据处理器
用于处理从预测服务接收到的 JSON 数据，提取目标标的和特征数据
"""

import json
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Tuple
from datetime import datetime


class PredictionDataProcessor:
    """
    预测数据处理器

    功能:
    1. 解析输入的 JSON 数据
    2. 提取目标预测标的 (TAR_INDEX_CODE)
    3. 提取目标标的的历史数据 (DATA_VALUE, DATA_DATE)
    4. 提取所有特征的历史数据
    """

    def __init__(self, data_source: Union[str, Path, dict]):
        """
        初始化数据处理器

        Parameters:
        -----------
        data_source : str, Path, or dict
            数据源，可以是:
            - JSON 文件路径 (str 或 Path)
            - 已加载的字典数据
        """
        self.raw_data = self._load_data(data_source)
        self.target_index_code = None
        self.target_data = None
        self.all_features_data = None
        self.metadata = None  # 保存元数据信息
        self.input_alg_id = None  # 保存输入的ALG_ID（如果有）

        # 自动处理数据
        self._process_data()
        pass

    def _load_data(self, data_source: Union[str, Path, dict]) -> dict:
        """加载 JSON 数据"""
        if isinstance(data_source, dict):
            return data_source

        # 文件路径
        file_path = Path(data_source)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _normalize_field_name(self, field_name: str) -> str:
        """
        标准化字段名
        将 dATA_DATE, iNDEX_CODE 等转换为 DATA_DATE, INDEX_CODE
        """
        return field_name.upper()

    def _extract_record_value(self, record: dict, field_name: str):
        """从记录中提取字段值，兼容不同的命名方式"""
        # 尝试原始字段名
        if field_name in record:
            return record[field_name]

        # 尝试小写开头的驼峰命名
        alt_name = field_name[0].lower() + field_name[1:]
        if alt_name in record:
            return record[alt_name]

        return None

    def _process_data(self):
        """处理数据，提取目标标的和特征数据"""
        # 提取目标标的代码
        self.target_index_code = self.raw_data.get('TAR_INDEX_CODE')
        if not self.target_index_code:
            raise ValueError("输入数据中缺少 TAR_INDEX_CODE 字段")

        # 提取 ALG_ID（如果存在）
        self.input_alg_id = self.raw_data.get('ALG_ID')

        # 提取 PRE_FEATURE_INFO
        pre_feature_info = self.raw_data.get('PRE_FEATURE_INFO', [])
        if not pre_feature_info:
            raise ValueError("输入数据中缺少 PRE_FEATURE_INFO 字段或数据为空")

        # 处理所有特征数据
        self._process_all_features(pre_feature_info)

        # 提取目标标的数据
        self._extract_target_data()

    def _process_all_features(self, pre_feature_info: List[dict]):
        """处理所有特征数据，转换为 DataFrame"""
        processed_records = []

        for record in pre_feature_info:
            processed_record = {
                'INDEX_CODE': self._extract_record_value(record, 'INDEX_CODE'),
                'INDEX_NAME': self._extract_record_value(record, 'INDEX_NAME'),
                'DATA_DATE': self._extract_record_value(record, 'DATA_DATE'),
                'DATA_VALUE': self._extract_record_value(record, 'DATA_VALUE'),
                'FREQUENCY': self._extract_record_value(record, 'FREQUENCY'),
            }
            processed_records.append(processed_record)

        # 转换为 DataFrame
        df = pd.DataFrame(processed_records)

        # 转换数据类型
        df['DATA_DATE'] = pd.to_datetime(df['DATA_DATE'])
        df['DATA_VALUE'] = pd.to_numeric(df['DATA_VALUE'], errors='coerce')

        # 按日期排序
        df = df.sort_values('DATA_DATE')

        self.all_features_data = df

    def _extract_target_data(self):
        """提取目标标的的数据，只保留三列：INDEX_CODE, DATA_DATE, DATA_VALUE"""
        if self.all_features_data is None:
            raise ValueError("尚未处理特征数据")

        # 筛选目标标的的数据
        target_df = self.all_features_data[
            self.all_features_data['INDEX_CODE'] == self.target_index_code
        ].copy()

        # 按日期排序
        target_df = target_df.sort_values('DATA_DATE').reset_index(drop=True)

        # 只保留三列
        self.target_data = target_df[['INDEX_CODE', 'DATA_DATE', 'DATA_VALUE']].copy()

        # 提取元数据信息
        self._extract_metadata(target_df)

    def _extract_metadata(self, target_df: pd.DataFrame):
        """
        提取元数据信息，用于输出时使用

        提取的信息包括:
        - INDEX_CODE: 标的代码
        - INDEX_NAME: 标的名称（可作为 MTRL_NAME）
        - FREQUENCY: 频率
        - MTRL_NO: 物料编号（需要外部提供或映射）
        """
        if len(target_df) == 0:
            raise ValueError("目标数据为空，无法提取元数据")

        # 从第一条记录中提取元数据（假设所有记录的元数据相同）
        first_record = target_df.iloc[0]

        self.metadata = {
            'INDEX_CODE': self.target_index_code,
            'INDEX_NAME': first_record.get('INDEX_NAME', ''),
            'FREQUENCY': first_record.get('FREQUENCY', 'D'),
            'MTRL_NO': None,  # 需要外部提供
            'MTRL_NAME': first_record.get('INDEX_NAME', ''),
        }

    def get_target_index_code(self) -> str:
        """
        获取目标预测标的代码

        Returns:
        --------
        str : 目标标的代码
        """
        return self.target_index_code

    def get_target_series(self) -> Tuple[pd.Series, pd.Series]:
        """
        获取目标标的的时间序列数据

        Returns:
        --------
        tuple : (dates, values)
            dates : pd.Series - 日期序列
            values : pd.Series - 数据值序列
        """
        if self.target_data is None:
            raise ValueError("目标数据未提取")

        dates = self.target_data['DATA_DATE']
        values = self.target_data['DATA_VALUE']

        return dates, values

    def get_target_dataframe(self) -> pd.DataFrame:
        """
        获取目标标的的完整 DataFrame (只包含 INDEX_CODE, DATA_DATE, DATA_VALUE 三列)

        Returns:
        --------
        pd.DataFrame : 包含 INDEX_CODE, DATA_DATE, DATA_VALUE 三列的 DataFrame
        """
        if self.target_data is None:
            raise ValueError("目标数据未提取")

        return self.target_data.copy()

    def get_metadata(self) -> Dict:
        """
        获取元数据信息

        Returns:
        --------
        dict : 包含以下字段的字典
            - INDEX_CODE: 标的代码
            - INDEX_NAME: 标的名称
            - FREQUENCY: 频率
            - MTRL_NO: 物料编号
            - MTRL_NAME: 物料名称
        """
        if self.metadata is None:
            raise ValueError("元数据未提取")

        return self.metadata.copy()

    def set_mtrl_no(self, mtrl_no: str):
        """
        设置物料编号（MTRL_NO）

        Parameters:
        -----------
        mtrl_no : str
            物料编号
        """
        if self.metadata is None:
            raise ValueError("元数据未提取")

        self.metadata['MTRL_NO'] = mtrl_no

    def generate_output_format(
        self,
        predictions: List[Dict[str, any]],
        alg_id: str = None,
        run_time: str = None,
        start_time: str = None,
        end_time: str = None
    ) -> Dict:
        """
        生成标准输出格式

        Parameters:
        -----------
        predictions : list of dict
            预测结果列表，每个元素包含:
            - PRICE_DATE: 预测日期
            - PRICE_VALUE: 预测价格
            - PRICE_VALUE_UB: 预测上界（可选）
            - PRICE_VALUE_LB: 预测下界（可选）
        alg_id : str, optional
            算法ID，如果不提供则自动生成
        run_time : str, optional
            运行时间，如果不提供则使用当前时间
        start_time : str, optional
            开始处理时间
        end_time : str, optional
            结束处理时间

        Returns:
        --------
        dict : 标准输出格式的字典
        """
        if self.metadata is None:
            raise ValueError("元数据未提取")

        # 生成 ALG_ID（如果未提供）
        if alg_id is None:
            # 优先使用输入的ALG_ID
            if self.input_alg_id:
                alg_id = self.input_alg_id
            else:
                from datetime import datetime
                alg_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # 生成 RUN_TIME（如果未提供）
        if run_time is None:
            from datetime import datetime
            run_time = datetime.now().isoformat() + "Z"

        # 构建输出记录
        output_records = []
        for pred in predictions:
            record = {
                'ALG_ID': alg_id,
                'FREQUENCY': self.metadata['FREQUENCY'],
                'MTRL_NO': self.metadata['MTRL_NO'],
                'MTRL_NAME': self.metadata['MTRL_NAME'],
                'INDEX_CODE': self.metadata['INDEX_CODE'],
                'RUN_TIME': run_time,
                'PRICE_DATE': pred['PRICE_DATE'],
                'PRICE_VALUE': pred['PRICE_VALUE'],
                'PRICE_VALUE_UB': pred.get('PRICE_VALUE_UB', None),
                'PRICE_VALUE_LB': pred.get('PRICE_VALUE_LB', None),
                'START_TIME': start_time,
                'END_TIME': end_time,
            }
            output_records.append(record)

        # 构建完整输出
        output = {
            'response_metadata': {
                'request_id': f"PRED_{alg_id}",
                'response_time': run_time,
                'status': 'success',
                'error_code': None,
                'error_message': None
            },
            'PRE_MATERIAL_PRICE_OUTPUT': output_records
        }

        return output

    def get_feature_series(self, feature_code: str) -> Tuple[pd.Series, pd.Series]:
        """
        获取指定特征的时间序列数据

        Parameters:
        -----------
        feature_code : str
            特征代码 (INDEX_CODE)

        Returns:
        --------
        tuple : (dates, values)
            dates : pd.Series - 日期序列
            values : pd.Series - 数据值序列
        """
        if self.all_features_data is None:
            raise ValueError("特征数据未提取")

        feature_df = self.all_features_data[
            self.all_features_data['INDEX_CODE'] == feature_code
        ].copy()

        feature_df = feature_df.sort_values('DATA_DATE').reset_index(drop=True)

        return feature_df['DATA_DATE'], feature_df['DATA_VALUE']

    def get_all_features_pivot(self) -> pd.DataFrame:
        """
        获取所有特征的宽格式 DataFrame (pivot 格式)
        行: 日期, 列: 特征代码, 值: DATA_VALUE

        Returns:
        --------
        pd.DataFrame : 宽格式的特征数据
        """
        if self.all_features_data is None:
            raise ValueError("特征数据未提取")

        pivot_df = self.all_features_data.pivot_table(
            index='DATA_DATE',
            columns='INDEX_CODE',
            values='DATA_VALUE',
            aggfunc='first'  # 如果有重复，取第一个值
        )

        pivot_df = pivot_df.sort_index()

        return pivot_df

    def get_available_features(self) -> List[str]:
        """
        获取所有可用的特征代码列表

        Returns:
        --------
        list : 特征代码列表
        """
        if self.all_features_data is None:
            raise ValueError("特征数据未提取")

        return sorted(self.all_features_data['INDEX_CODE'].unique().tolist())

    def get_data_summary(self) -> Dict:
        """
        获取数据摘要信息

        Returns:
        --------
        dict : 包含数据统计信息的字典
        """
        if self.all_features_data is None or self.target_data is None:
            raise ValueError("数据未处理")

        summary = {
            'target_index_code': self.target_index_code,
            'target_data_points': len(self.target_data),
            'target_date_range': {
                'start': self.target_data['DATA_DATE'].min().strftime('%Y-%m-%d'),
                'end': self.target_data['DATA_DATE'].max().strftime('%Y-%m-%d')
            },
            'total_features': len(self.get_available_features()),
            'feature_codes': self.get_available_features(),
            'total_records': len(self.all_features_data)
        }

        return summary

    def compute_lagged_correlations(self, lag_days: int = 30) -> pd.DataFrame:
        """
        计算所有特征变量（前置lag_days天后）与目标变量的相关性

        Parameters:
        -----------
        lag_days : int, default=30
            前置天数（将数据后移的天数）

        Returns:
        --------
        pd.DataFrame : 包含以下列的 DataFrame
            - INDEX_CODE: 特征变量代码
            - INDEX_NAME: 特征变量名称
            - CORRELATION: 相关系数
            - CHANGE_RATE: 特征变化率（使用后移后的数据计算）
        """
        if self.all_features_data is None or self.target_data is None:
            raise ValueError("数据未处理")

        # 获取目标变量的时间序列
        # 去除重复的日期，保留最后一个值
        target_df_unique = self.target_data.drop_duplicates(subset=['DATA_DATE'], keep='last')
        target_series = target_df_unique.set_index('DATA_DATE')['DATA_VALUE'].sort_index()

        # 获取所有非目标变量的特征
        feature_codes = [
            code for code in self.get_available_features()
            if code != self.target_index_code
        ]

        results = []

        for feature_code in feature_codes:
            # 获取特征变量的时间序列
            feature_df = self.all_features_data[
                self.all_features_data['INDEX_CODE'] == feature_code
            ].copy()

            if len(feature_df) == 0:
                continue

            feature_df = feature_df.sort_values('DATA_DATE').reset_index(drop=True)

            # 去除重复的日期，保留最后一个值
            feature_df = feature_df.drop_duplicates(subset=['DATA_DATE'], keep='last')

            feature_series = feature_df.set_index('DATA_DATE')['DATA_VALUE'].sort_index()

            # 获取特征名称
            feature_name = feature_df['INDEX_NAME'].iloc[0] if 'INDEX_NAME' in feature_df.columns else feature_code

            # 后移 lag_days 天
            feature_series_lagged = feature_series.shift(-lag_days)

            # 对齐两个时间序列（只保留共同的日期）
            common_dates = target_series.index.intersection(feature_series_lagged.index)

            if len(common_dates) < 10:  # 至少需要10个数据点才能计算相关性
                continue

            target_aligned = target_series.loc[common_dates]
            feature_aligned = feature_series_lagged.loc[common_dates]

            # 去除 NaN 值
            valid_mask = target_aligned.notna() & feature_aligned.notna()
            target_valid = target_aligned[valid_mask]
            feature_valid = feature_aligned[valid_mask]

            if len(target_valid) < 10:
                continue

            # 计算相关系数
            correlation = target_valid.corr(feature_valid)

            # 计算特征变化率（使用后移后的数据）
            # 取最后两个有效的数据点
            if len(feature_valid) >= 2:
                last_value = feature_valid.iloc[-1]
                prev_value = feature_valid.iloc[-2]
                if prev_value != 0 and not pd.isna(prev_value) and not pd.isna(last_value):
                    change_rate = (last_value - prev_value) / prev_value
                else:
                    change_rate = 0.0
            else:
                change_rate = 0.0

            results.append({
                'INDEX_CODE': feature_code,
                'INDEX_NAME': feature_name,
                'CORRELATION': correlation if not pd.isna(correlation) else 0.0,
                'CHANGE_RATE': change_rate
            })

        # 转换为 DataFrame
        results_df = pd.DataFrame(results)

        # 按相关系数的绝对值排序
        if len(results_df) > 0:
            results_df['ABS_CORRELATION'] = results_df['CORRELATION'].abs()
            results_df = results_df.sort_values('ABS_CORRELATION', ascending=False).reset_index(drop=True)
            results_df = results_df.drop(columns=['ABS_CORRELATION'])

        return results_df

    def generate_character_factor_output(
        self,
        correlations_df: pd.DataFrame,
        alg_id: str = None,
        run_time: str = None
    ) -> List[Dict]:
        """
        生成 PRE_CHARACTER_FACTOR_OUTPUT 表

        Parameters:
        -----------
        correlations_df : pd.DataFrame
            相关性分析结果（来自 compute_lagged_correlations）
        alg_id : str, optional
            算法ID
        run_time : str, optional
            运行时间

        Returns:
        --------
        list of dict : PRE_CHARACTER_FACTOR_OUTPUT 表的记录列表
        """
        if self.metadata is None:
            raise ValueError("元数据未提取")

        # 生成 ALG_ID（如果未提供）
        if alg_id is None:
            # 优先使用输入的ALG_ID
            if self.input_alg_id:
                alg_id = self.input_alg_id
            else:
                alg_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # 生成 CALCULATE_TIME（如果未提供）
        if run_time is None:
            run_time = datetime.now().isoformat() + "Z"

        output_records = []

        for _, row in correlations_df.iterrows():
            record = {
                'ALG_ID': alg_id,
                'FREQUENCY': self.metadata['FREQUENCY'],
                'MTRL_NO': self.metadata['MTRL_NO'],
                'MTRL_NAME': self.metadata['MTRL_NAME'],
                'INDEX_CODE': self.metadata['INDEX_CODE'],
                'CALCULATE_TIME': run_time,
                'CHARACTER_FACTOR': row['INDEX_NAME'],
                'WEIGHT_COEFFICIENT': round(row['CORRELATION'], 6),
                'CHARACTER_CHANGE_RATE': round(row['CHANGE_RATE'], 6)
            }
            output_records.append(record)

        return output_records

    def generate_influence_factor_output(
        self,
        correlations_df: pd.DataFrame,
        alg_id: str = None,
        run_time: str = None,
        top_n: int = 3
    ) -> List[Dict]:
        """
        生成 PRE_INFLUENCE_FACTOR_OUTPUT 表

        Parameters:
        -----------
        correlations_df : pd.DataFrame
            相关性分析结果（来自 compute_lagged_correlations）
        alg_id : str, optional
            算法ID
        run_time : str, optional
            运行时间
        top_n : int, default=3
            取前 N 个利好/利空因素

        Returns:
        --------
        list of dict : PRE_INFLUENCE_FACTOR_OUTPUT 表的记录列表（只有一条记录）
        """
        if self.metadata is None:
            raise ValueError("元数据未提取")

        # 生成 ALG_ID（如果未提供）
        if alg_id is None:
            # 优先使用输入的ALG_ID
            if self.input_alg_id:
                alg_id = self.input_alg_id
            else:
                alg_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # 生成 CALCULATE_TIME（如果未提供）
        if run_time is None:
            run_time = datetime.now().isoformat() + "Z"

        # 分离正负相关的因素
        positive_factors = correlations_df[correlations_df['CORRELATION'] > 0].copy()
        negative_factors = correlations_df[correlations_df['CORRELATION'] < 0].copy()

        # 按相关系数绝对值排序，取前 top_n 个
        positive_factors = positive_factors.nlargest(top_n, 'CORRELATION')
        negative_factors = negative_factors.nsmallest(top_n, 'CORRELATION')

        # 拼接因素名称（用分号分隔）
        positive_names = ';'.join(positive_factors['INDEX_NAME'].tolist())
        negative_names = ';'.join(negative_factors['INDEX_NAME'].tolist())

        # 构建记录
        record = {
            'ALG_ID': alg_id,
            'FREQUENCY': self.metadata['FREQUENCY'],
            'MTRL_NO': self.metadata['MTRL_NO'],
            'MTRL_NAME': self.metadata['MTRL_NAME'],
            'INDEX_CODE': self.metadata['INDEX_CODE'],
            'CALCULATE_TIME': run_time,
            'NEGATIVE_FACTORS': negative_names,
            'POSITIVE_FACTORS': positive_names
        }

        return [record]

    def save_target_to_csv(self, output_path: Union[str, Path]):
        """
        将目标标的数据保存为 CSV 文件

        Parameters:
        -----------
        output_path : str or Path
            输出文件路径
        """
        if self.target_data is None:
            raise ValueError("目标数据未提取")

        self.target_data.to_csv(output_path, index=False, encoding='utf-8')
        print(f"目标数据已保存到: {output_path}")

    def save_all_features_to_csv(self, output_path: Union[str, Path], pivot: bool = True):
        """
        将所有特征数据保存为 CSV 文件

        Parameters:
        -----------
        output_path : str or Path
            输出文件路径
        pivot : bool, default=True
            是否保存为宽格式 (pivot)，如果为 False，保存为长格式
        """
        if self.all_features_data is None:
            raise ValueError("特征数据未提取")

        if pivot:
            df = self.get_all_features_pivot()
        else:
            df = self.all_features_data

        df.to_csv(output_path, index=pivot, encoding='utf-8')
        print(f"所有特征数据已保存到: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 示例 1: 从 JSON 文件加载
    json_file = Path(__file__).parent / "input_20251225_175837291446.json"

    if json_file.exists():
        print("=" * 60)
        print("示例: 处理预测数据")
        print("=" * 60)

        # 创建数据处理器
        processor = PredictionDataProcessor(json_file)

        # 获取数据摘要
        summary = processor.get_data_summary()
        print("\n数据摘要:")
        print(f"  目标标的代码: {summary['target_index_code']}")
        print(f"  目标数据点数: {summary['target_data_points']}")
        print(f"  数据日期范围: {summary['target_date_range']['start']} 至 {summary['target_date_range']['end']}")
        print(f"  可用特征数量: {summary['total_features']}")
        print(f"  总记录数: {summary['total_records']}")

        # 获取目标序列
        dates, values = processor.get_target_series()
        print(f"\n目标序列示例 (前 5 条):")
        for i in range(min(5, len(dates))):
            print(f"  {dates.iloc[i].strftime('%Y-%m-%d')}: {values.iloc[i]}")

        # 获取所有特征列表
        print(f"\n可用特征:")
        for feature in summary['feature_codes']:
            count = len(processor.all_features_data[
                processor.all_features_data['INDEX_CODE'] == feature
            ])
            print(f"  - {feature}: {count} 条记录")

        # # 保存目标数据
        # output_dir = Path(__file__).parent / "processed_data"
        # output_dir.mkdir(exist_ok=True)
        #
        # target_csv = output_dir / f"{summary['target_index_code']}_target.csv"
        # processor.save_target_to_csv(target_csv)
        #
        # # 保存所有特征数据 (宽格式)
        # all_features_csv = output_dir / f"{summary['target_index_code']}_all_features_pivot.csv"
        # processor.save_all_features_to_csv(all_features_csv, pivot=True)

        print(f"\n处理完成!")
        print("=" * 60)
    else:
        print(f"示例 JSON 文件不存在: {json_file}")
