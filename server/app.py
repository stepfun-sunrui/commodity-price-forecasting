"""
价格预测服务

使用配置驱动的预测服务，从src目录读取配置
"""
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import sys
import json
import datetime
import logging

# 添加项目根目录到路径
server_dir = Path(__file__).parent
project_root = server_dir.parent
sys.path.insert(0, str(server_dir))
sys.path.insert(0, str(project_root))

from predict import PricePrediction
from src.core.config_manager import ConfigManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="价格预测服务",
    description="商品价格预测 API，使用配置驱动的预测",
    version="2.0.0"
)

# 物料编号映射
MTRL_NO_MAPPING = {
    "RE00035675": "COAL_RE00035675",
    "ID00102866": "IRON_ID00102866",
    "ID00103568": "MTRL_ID00103568",
    "ID00103617": "MTRL_ID00103617",
    "ID01020441": "MTRL_ID01020441",
    "ID01560197": "MTRL_ID01560197",
}

# 确保目录存在
data_dir = Path(__file__).parent / "data"
output_dir = Path(__file__).parent / "output"
data_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

# 回调接口配置
CALLBACK_URL = "https://yzpms.yong-gang.cn/prod-api/yfl/api/commonApi/receiveMaterialData"
CALLBACK_TIMEOUT = 30


def send_prediction_callback(prediction_data: dict) -> bool:
    """
    将预测结果发送到回调接口

    Parameters:
    -----------
    prediction_data : dict
        预测结果数据

    Returns:
    --------
    bool : 是否发送成功
    """
    import requests

    try:
        logger.info(f"开始发送预测结果到回调接口: {CALLBACK_URL}")

        response = requests.post(
            CALLBACK_URL,
            json=prediction_data,
            timeout=CALLBACK_TIMEOUT,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            logger.info(f"回调接口调用成功，状态码: {response.status_code}")
            return True
        else:
            logger.warning(f"回调接口返回非200状态码: {response.status_code}, 响应: {response.text}")
            return False

    except requests.exceptions.Timeout:
        logger.error(f"回调接口超时（{CALLBACK_TIMEOUT}秒）")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"回调接口调用失败: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"发送回调时发生未知错误: {str(e)}", exc_info=True)
        return False


def execute_prediction_task(
    data: dict,
    target_codes: list,
    alg_id: str,
    timestamp: str
):
    """
    后台任务：顺序执行多个标的的预测并发送回调

    Parameters:
    -----------
    data : dict
        输入数据
    target_codes : list
        目标标的代码列表
    alg_id : str
        算法ID
    timestamp : str
        时间戳
    """
    logger.info(f"后台任务开始：共需预测 {len(target_codes)} 个标的")

    success_count = 0
    fail_count = 0

    # 顺序处理每个标的
    for idx, target_code in enumerate(target_codes, 1):
        try:
            logger.info(f"[{idx}/{len(target_codes)}] 处理标的: {target_code}")

            # 从配置文件读取配置
            config = ConfigManager.load_target(target_code)

            # 获取模型类型和参数
            prediction_mode = config.model_type
            horizon_days = config.horizon_days
            n_predictions = config.n_predictions

            # 根据模型类型获取数据年限
            if prediction_mode == "lgbm":
                years = config.lgbm_config.get('data', {}).get('years', 5)
            elif prediction_mode == "sarima":
                years = config.sarima_config.get('data', {}).get('years', 5)
            else:
                years = 5

            min_train_samples = config.min_train_samples

            logger.info(f"配置: 模型={prediction_mode}, 年限={years}, 预测={n_predictions}天")

            # 设置物料编号
            mtrl_no = MTRL_NO_MAPPING.get(target_code, f"MTRL_{target_code}")

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

                logger.info(f"数据过滤：原始 {original_count} 条 -> 过滤后 {filtered_count} 条")

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
                run_time=datetime.datetime.now().isoformat() + "Z"
            )

            # 保存输出结果
            output_file = output_dir / f"prediction_{target_code}_{timestamp}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logger.info(f"预测结果已保存: {output_file}")

            # 发送预测结果到回调接口
            callback_success = send_prediction_callback(output)
            if callback_success:
                logger.info(f"{target_code} 预测结果已成功发送到回调接口")
            else:
                logger.warning(f"{target_code} 预测结果发送到回调接口失败")

            success_count += 1
            logger.info(f"[{idx}/{len(target_codes)}] {target_code} 预测成功")

        except Exception as e:
            fail_count += 1
            logger.error(f"[{idx}/{len(target_codes)}] {target_code} 处理异常: {str(e)}", exc_info=True)

    # 汇总结果
    logger.info(f"所有预测任务完成：成功 {success_count} 个，失败 {fail_count} 个")


@app.get("/")
async def root():
    """API 根路径"""
    return {
        "service": "价格预测服务（新架构）",
        "version": "2.0.0",
        "architecture": "unified",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "config": "/config",
            "model_info": "/model_info/{target_code}"
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "architecture": "new"
    }


@app.get("/config")
async def config():
    """返回当前配置"""
    return {
        "supported_targets": [
            "ID00102866", "ID00103568", "ID00103617",
            "ID01020441", "ID01560197", "RE00035675"
        ],
        "callback_url": CALLBACK_URL,
        "architecture": "unified"
    }


@app.get("/model_info/{target_code}")
async def model_info(target_code: str):
    """获取模型信息"""
    try:
        info = prediction_service.get_model_info(target_code)
        return {
            "status": "success",
            "data": info
        }
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "error_message": str(e)
            },
            status_code=404
        )


@app.post("/predict")
async def predict(request: Request, background_tasks: BackgroundTasks):
    """
    价格预测接口（异步处理）

    接收数据后立即返回响应，预测在后台执行，完成后通过回调接口返回结果
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    response_time = datetime.datetime.now().isoformat() + "Z"

    try:
        # 获取 JSON 数据
        data = await request.json()
        logger.info(f"接收到预测请求，时间戳: {timestamp}")

        # 验证必需字段
        if "TAR_INDEX_CODE" not in data:
            logger.error("缺少 TAR_INDEX_CODE 字段")
            return JSONResponse(
                content={
                    "response_metadata": {
                        "request_id": f"PRED_{timestamp}",
                        "response_time": response_time,
                        "status": "error",
                        "error_code": "MISSING_FIELD",
                        "error_message": "缺少 TAR_INDEX_CODE 字段"
                    }
                },
                status_code=400
            )

        if "PRE_FEATURE_INFO" not in data or not data["PRE_FEATURE_INFO"]:
            logger.error("缺少 PRE_FEATURE_INFO 字段或数据为空")
            return JSONResponse(
                content={
                    "response_metadata": {
                        "request_id": f"PRED_{timestamp}",
                        "response_time": response_time,
                        "status": "error",
                        "error_code": "MISSING_FIELD",
                        "error_message": "缺少 PRE_FEATURE_INFO 字段或数据为空"
                    }
                },
                status_code=400
            )

        # 保存输入数据
        timestamp2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = data_dir / f"input_{timestamp2}.json"
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"输入数据已保存: {input_file}")

        # 获取目标标的代码（支持字符串或列表）
        tar_index_code = data['TAR_INDEX_CODE']
        if isinstance(tar_index_code, str):
            target_codes = [tar_index_code]
            logger.info(f"目标标的（单个）: {tar_index_code}")
        elif isinstance(tar_index_code, list):
            target_codes = tar_index_code
            logger.info(f"目标标的（多个）: {target_codes}")
        else:
            logger.error(f"TAR_INDEX_CODE 格式错误")
            return JSONResponse(
                content={
                    "response_metadata": {
                        "request_id": f"PRED_{timestamp}",
                        "response_time": response_time,
                        "status": "error",
                        "error_code": "INVALID_FORMAT",
                        "error_message": "TAR_INDEX_CODE 应为字符串或列表"
                    }
                },
                status_code=400
            )

        # 验证标的列表不为空
        if not target_codes:
            logger.error("TAR_INDEX_CODE 列表为空")
            return JSONResponse(
                content={
                    "response_metadata": {
                        "request_id": f"PRED_{timestamp}",
                        "response_time": response_time,
                        "status": "error",
                        "error_code": "EMPTY_LIST",
                        "error_message": "TAR_INDEX_CODE 列表不能为空"
                    }
                },
                status_code=400
            )

        logger.info(f"共需预测 {len(target_codes)} 个标的")

        # 获取ALG_ID
        alg_id = data.get('ALG_ID', timestamp)
        if 'ALG_ID' in data:
            logger.info(f"使用输入数据中的 ALG_ID: {alg_id}")
        else:
            logger.info(f"输入数据中无 ALG_ID，使用生成的时间戳: {alg_id}")

        # 添加后台任务
        background_tasks.add_task(
            execute_prediction_task,
            data=data,
            target_codes=target_codes,
            alg_id=alg_id,
            timestamp=timestamp
        )

        logger.info("预测任务已添加到后台队列")

        # 立即返回响应
        return JSONResponse(
            content={
                "response_metadata": {
                    "request_id": f"PRED_{alg_id}",
                    "response_time": response_time,
                    "status": "accepted",
                    "message": f"预测任务已接收，正在后台顺序处理 {len(target_codes)} 个标的，每个完成后将通过回调接口返回结果"
                },
                "task_info": {
                    "target_codes": target_codes,
                    "target_count": len(target_codes),
                    "alg_id": alg_id,
                    "processing_mode": "sequential",
                    "architecture": "new"
                }
            },
            status_code=202
        )

    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析错误: {str(e)}")
        return JSONResponse(
            content={
                "response_metadata": {
                    "request_id": f"PRED_{timestamp}",
                    "response_time": response_time,
                    "status": "error",
                    "error_code": "INVALID_JSON",
                    "error_message": f"无效的 JSON 格式: {str(e)}"
                }
            },
            status_code=400
        )

    except Exception as e:
        logger.error(f"请求处理错误: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "response_metadata": {
                    "request_id": f"PRED_{timestamp}",
                    "response_time": response_time,
                    "status": "error",
                    "error_code": "REQUEST_ERROR",
                    "error_message": str(e)
                }
            },
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=6001, reload=False)
