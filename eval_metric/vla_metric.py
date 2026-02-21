import os
import json
import math
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  # 需要安装fastdtw库：pip install fastdtw

# 常量定义
THRESHOLD_STOP_DIST = 0.15
THRESHOLD_STOP_ANGLE = math.pi / 12
THRESHOLD_SUCCESS_DIST = 0.5
THRESHOLD_SUCCESS_ANGLE = math.pi / 4
ALPHA = 1  # NDTW参数

DATASET_ROOT = "/home/testunot/datasets/habitat/IndoorUAV-VLA"
SHARED_FOLDER = (
    "/home/testunot/IndoorUAV-Agent/online_eval/vla_eval/shared_folder"
)
TRAJECTORIES_DIR = os.path.join(SHARED_FOLDER, "trajectories")
OUTPUT_FILE = os.path.join(SHARED_FOLDER, "evaluation_openvla_results.json")


def angle_difference(a, b):
    """计算两个角度之间的最小差异（考虑圆周性）"""
    diff = abs(a - b)
    return min(diff, 2 * math.pi - diff)


def calculate_ndtw(seq_a, seq_b, is_angle=False):
    """
    计算两个序列之间的NDTW
    :param seq_a: 序列A
    :param seq_b: 序列B
    :param is_angle: 是否为角度序列
    :return: (NDTW值, 参考路径长度L)
    """
    if len(seq_a) == 0 or len(seq_b) == 0:
        return 0.0, 0.0

    arr_a = np.array(seq_a).reshape(-1, 1) if is_angle else np.array(seq_a)
    arr_b = np.array(seq_b).reshape(-1, 1) if is_angle else np.array(seq_b)
    dist_func = angle_difference if is_angle else euclidean

    # 计算累积距离
    distance, _ = fastdtw(arr_a, arr_b, dist=dist_func)

    # 计算参考路径长度L
    if is_angle:
        L = sum(
            angle_difference(seq_b[i], seq_b[i - 1])
            for i in range(1, len(seq_b))
        )
    else:
        L = sum(
            euclidean(seq_b[i], seq_b[i - 1]) for i in range(1, len(seq_b))
        )

    # 避免除以零
    L = max(L, 1e-5)

    # 计算NDTW
    ndtw_value = math.exp(-distance / (ALPHA * L))

    return ndtw_value, L


def process_episode(trajectory_file):
    """处理单个episode文件"""
    try:
        # 加载轨迹文件
        with open(trajectory_file, "r", encoding="gbk") as f:
            traj_data = json.load(f)

        # 获取episode_key和轨迹
        episode_key = traj_data["episode_key"].lstrip("/")
        trajectory = traj_data["trajectory"]

        # 解析路径
        parts = episode_key.split("/")
        scene_name, env_name, traj_folder, vla_file = parts[:4]

        # 构建vla_ins文件路径
        vla_ins_path = os.path.join(
            DATASET_ROOT, "vla_ins", scene_name, env_name, traj_folder, vla_file
        )
        with open(vla_ins_path, "r", encoding="gbk") as f:
            vla_data = json.load(f)
        source = vla_data["source"]

        # 构建posture文件路径
        posture_path = os.path.join(
            DATASET_ROOT,
            "without_screenshot",
            scene_name,
            env_name,
            traj_folder,
            "posture.json",
        )
        with open(posture_path, "r") as f:
            posture_data = json.load(f)

        # 提取gt序列 (0-based索引)
        start_idx = source[0] - 1
        end_idx = source[1] - 1
        gt_full_seq = posture_data[start_idx : end_idx + 1]

        # 转换角度为弧度
        gt_seq = []
        for point in gt_full_seq:
            x, y, z, yaw_deg = point
            yaw_rad = yaw_deg * math.pi / 180.0
            gt_seq.append([x, y, z, yaw_rad])

        # 处理预测轨迹 (跳过第0个点)
        pred_full_seq = trajectory[2:16]  # 取第1到15个点

        if len(pred_full_seq) == 0:
            print(
                f"Skipping {trajectory_file}: trajectory too short "
                f"({len(trajectory)} points)"
            )
            return None

        # 检查是否满足停止条件
        stop_index = None
        for i in range(1, len(pred_full_seq)):
            prev = pred_full_seq[i - 1]
            curr = pred_full_seq[i]

            # 计算距离和角度差
            dist = euclidean(prev[:3], curr[:3])
            angle_diff = angle_difference(prev[3], curr[3])

            if dist < THRESHOLD_STOP_DIST and \
               angle_diff < THRESHOLD_STOP_ANGLE:
                stop_index = i - 1  # 使用前一个点作为终点
                break

        # 确定最终使用的预测序列
        if stop_index is not None:
            pred_seq = pred_full_seq[: stop_index + 1]  # 包含停止点
        else:
            pred_seq = pred_full_seq  # 使用完整序列

        # 获取最后一个预测点和gt点
        last_pred = pred_seq[-1]
        last_gt = gt_seq[-1]

        # 计算最终距离和角度差
        final_dist = euclidean(last_pred[:3], last_gt[:3])
        final_angle_diff = angle_difference(last_pred[3], last_gt[3])

        # 判断是否成功
        success = (
            final_dist < THRESHOLD_SUCCESS_DIST
            and final_angle_diff < THRESHOLD_SUCCESS_ANGLE
        )

        # 准备NDTW计算
        pred_positions = [p[:3] for p in pred_seq]
        pred_angles = [p[3] for p in pred_seq]

        gt_positions = [p[:3] for p in gt_seq]
        gt_angles = [p[3] for p in gt_seq]

        # 计算位置NDTW和参考长度
        nDTW_pos, L_pos = calculate_ndtw(pred_positions, gt_positions)

        # 计算角度NDTW和参考长度
        nDTW_ang, L_ang = calculate_ndtw(pred_angles, gt_angles, is_angle=True)

        L_pos_adjusted = L_pos / 2.2

        # 计算权重 (避免除零错误)
        total_adjusted = L_pos_adjusted + L_ang
        if total_adjusted > 0:
            weight_pos = L_pos_adjusted / total_adjusted
            weight_ang = L_ang / total_adjusted
        else:
            weight_pos = 0.5
            weight_ang = 0.5

        # 计算综合NDTW
        nDTW_total = weight_pos * nDTW_pos + weight_ang * nDTW_ang

        result_dict = {
            "episode": episode_key,
            "success": success,
            "nDTW": nDTW_total if stop_index is not None else None,
            "final_dist": final_dist,
            "final_angle_diff": final_angle_diff,
        }

        return result_dict
    except Exception as e:
        print(f"Error processing {trajectory_file}: {str(e)}")
        return None


def get_base_stats_dict():
    """返回用于统计的基础数据字典结构"""
    return {
        "success_count": 0,
        "total_count": 0,
        "ndtw_count": 0,
        "total_nDTW": 0.0,
    }


def main():
    # 配置路径
    trajectories_dir = TRAJECTORIES_DIR
    output_file = OUTPUT_FILE
    final_results_file = os.path.join(trajectories_dir, "final_results.json")

    # 提取difficulty映射表
    difficulty_map = {}
    if os.path.exists(final_results_file):
        with open(final_results_file, "r") as f:
            final_results_data = json.load(f)
            for key, value in final_results_data.items():
                # 去除前导斜杠，确保与process_episode输出的key格式一致
                normalized_key = key.lstrip("/")
                difficulty_map[normalized_key] = value.get("difficulty", "unknown")
    else:
        print(f"Warning: {final_results_file} not found. "
              "All metrics will be grouped as 'unknown'.")

    # 初始化统计数据结构
    metrics = {
        "overall": get_base_stats_dict(),
        "easy": get_base_stats_dict(),
        "medium": get_base_stats_dict(),
        "hard": get_base_stats_dict(),
        "unknown": get_base_stats_dict(),
    }

    all_results = []

    # 遍历所有轨迹文件
    for filename in os.listdir(trajectories_dir):
        if not filename.endswith(".json") or filename == "final_results.json":
            continue

        filepath = os.path.join(trajectories_dir, filename)
        result = process_episode(filepath)

        if result:
            episode_key = result["episode"]
            difficulty = difficulty_map.get(episode_key, "unknown")
            result["difficulty"] = difficulty
            all_results.append(result)

            # 数据清洗：限定nDTW范围在[0, 1]内
            if result["nDTW"] is not None and result["nDTW"] > 1:
                result["nDTW"] = None

            # 需要更新的类别：overall 和 具体的difficulty级别
            categories_to_update = ["overall", difficulty]

            for cat in categories_to_update:
                metrics[cat]["total_count"] += 1

                if result["success"]:
                    metrics[cat]["success_count"] += 1

                if result["nDTW"] is not None and result["nDTW"] < 1:
                    metrics[cat]["ndtw_count"] += 1
                    metrics[cat]["total_nDTW"] += result["nDTW"]

    # 计算结果及输出字典构建
    final_metrics_output = {}

    for cat, stats in metrics.items():
        if stats["total_count"] == 0:
            continue  # 跳过没有数据的分类（例如unknown）

        success_rate = (
            stats["success_count"] / stats["total_count"]
            if stats["total_count"] > 0
            else 0.0
        )
        avg_nDTW = (
            stats["total_nDTW"] / stats["ndtw_count"]
            if stats["ndtw_count"] > 0
            else 0.0
        )

        final_metrics_output[cat] = {
            "success_rate": success_rate,
            "average_nDTW": avg_nDTW,
            "total_episodes": stats["total_count"],
            "success_count": stats["success_count"],
            "ndtw_valid_count": stats["ndtw_count"]
        }

    # 保存结果
    with open(output_file, "w") as f:
        json.dump(
            {
                "metrics_by_difficulty": final_metrics_output,
                "per_episode_results": all_results,
            },
            f,
            indent=2,
        )

    # 打印结果
    print("=== Evaluation Results ===")
    for cat, data in final_metrics_output.items():
        print(f"\n[{cat.upper()}]")
        print(f"  Total episodes: {data['total_episodes']}")
        print(f"  Success rate:   {data['success_rate']:.4f} "
              f"({data['success_count']}/{data['total_episodes']})")
        print(f"  Average nDTW:   {data['average_nDTW']:.4f} "
              f"(Based on {data['ndtw_valid_count']} valid nDTW values)")

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()