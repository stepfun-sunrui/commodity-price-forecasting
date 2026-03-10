"""
测试预测服务

直接读取JSON文件并运行预测，保存结果到output目录
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
server_dir = Path(__file__).parent
project_root = server_dir.parent
sys.path.insert(0, str(server_dir))
sys.path.insert(0, str(project_root))

from predict import PricePrediction
from src.core.config_manager import ConfigManager


def test_prediction_from_json(json_file: str):
    """
    从JSON文件读取数据并执行预测

    Parameters:
    -----------
    json_file : str
        输入JSON文件路径
    """
    # 读取JSON文件
    json_path = Path(json_file)
    if not json_path.exists():
        print(f"错误: 文件不存在 {json_path}")
        return

    print(f"读取输入文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取目标标的代码
    tar_index_code = data.get('TAR_INDEX_CODE')
    if not tar_index_code:
        print("错误: 缺少 TAR_INDEX_CODE 字段")
        return

    # 支持单个或多个标的
    if isinstance(tar_index_code, str):
        target_codes = [tar_index_code]
    elif isinstance(tar_index_code, list):
        target_codes = tar_index_code
    else:
        print("错误: TAR_INDEX_CODE 格式错误")
        return

    print(f"目标标的: {target_codes}")
    print(f"共需预测 {len(target_codes)} 个标的\n")

    # 确保输出目录存在
    output_dir = server_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # 获取ALG_ID
    alg_id = data.get('ALG_ID', datetime.now().strftime("%Y%m%d%H%M%S"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 物料编号映射
    mtrl_no_mapping = {
        "RE00035675": "COAL_RE00035675",
        "ID00102866": "IRON_ID00102866",
        "ID00103568": "MTRL_ID00103568",
        "ID00103617": "MTRL_ID00103617",
        "ID01020441": "MTRL_ID01020441",
        "ID01560197": "MTRL_ID01560197",
    }

    # 顺序处理每个标的
    success_count = 0
    fail_count = 0

    for idx, target_code in enumerate(target_codes, 1):
        try:
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(target_codes)}] 处理标的: {target_code}")
            print(f"{'='*60}")

            # 从配置文件读取配置
            config = ConfigManager.load_target(target_code)

            # 获取模型类型
            prediction_mode = config.model_type

            # 获取预测参数
            horizon_days = config.horizon_days
            n_predictions = config.n_predictions

            # 根据模型类型获取数据年限
            if prediction_mode == "lgbm":
                years = config.lgbm_config.get('data', {}).get('years', 5)
            elif prediction_mode == "sarima":
                years = config.sarima_config.get('data', {}).get('years', 5)
            else:
                years = 5

            # 最小训练样本数
            min_train_samples = config.min_train_samples

            print(f"配置信息:")
            print(f"  - 模型类型: {prediction_mode}")
            print(f"  - 数据年限: {years} 年")
            print(f"  - 预测时域: {horizon_days} 天")
            print(f"  - 预测点数: {n_predictions} 个")
            print(f"  - 最小训练样本: {min_train_samples}")

            # 设置物料编号
            mtrl_no = mtrl_no_mapping.get(target_code, f"MTRL_{target_code}")
            print(f"  - 物料编号: {mtrl_no}")

            # 创建数据副本，过滤掉其他目标标的的数据
            data_filtered = data.copy()

            # 过滤 PRE_FEATURE_INFO
            if 'PRE_FEATURE_INFO' in data_filtered:
                original_count = len(data_filtered['PRE_FEATURE_INFO'])

                # 保留当前标的的数据 + 不在目标列表中的特征数据
                filtered_features = [
                    record for record in data_filtered['PRE_FEATURE_INFO']
                    if record.get('INDEX_CODE') == target_code or record.get('INDEX_CODE') not in target_codes
                ]

                data_filtered['PRE_FEATURE_INFO'] = filtered_features
                filtered_count = len(filtered_features)

                print(f"数据过滤：原始 {original_count} 条 -> 过滤后 {filtered_count} 条")

            # 设置当前标的
            data_filtered['TAR_INDEX_CODE'] = target_code

            # 创建预测器
            predictor = PricePrediction(data_filtered)
            predictor.set_mtrl_no(mtrl_no)

            # 执行预测
            output = predictor.predict_and_generate_output(
                horizon_days=horizon_days,
                years=years,
                min_train_samples=min_train_samples,
                n_predictions=n_predictions,
                prediction_mode=prediction_mode,
                alg_id=alg_id,
                run_time=datetime.now().isoformat() + "Z"
            )

            # 保存输出结果
            output_file = output_dir / f"prediction_{target_code}_{timestamp}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f"\n预测结果已保存: {output_file}")
            print(f"预测点数: {len(output['PRE_MATERIAL_PRICE_OUTPUT'])}")

            success_count += 1

        except Exception as e:
            fail_count += 1
            print(f"\n错误: {target_code} 处理失败")
            print(f"异常信息: {str(e)}")
            import traceback
            traceback.print_exc()

    # 汇总结果
    print(f"\n{'='*60}")
    print(f"所有预测任务完成")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 默认使用 server 目录下的 input_20260130_155428.json
    default_json = server_dir / "input_20260130_155428.json"

    if len(sys.argv) > 1:
        # 如果提供了命令行参数，使用指定的文件
        json_file = sys.argv[1]
    else:
        # 否则使用默认文件
        json_file = str(default_json)

    print(f"测试预测服务")
    print(f"输入文件: {json_file}\n")

    test_prediction_from_json(json_file)
