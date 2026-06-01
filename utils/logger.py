import csv
import os
from datetime import datetime

def log_experiment_to_csv(csv_path, args_dict, metrics_dict):
    """
    将实验的超参数和最终指标规范化写入 CSV 文件。
    这种字典传参的方式极其灵活，以后加减参数都不需要改核心逻辑。
    """
    file_exists = os.path.isfile(csv_path)
    
    # 动态生成表头和数据行
    headers = list(args_dict.keys()) + list(metrics_dict.keys())
    values = list(args_dict.values()) + list(metrics_dict.values())
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 只有在文件第一次创建时才写入表头
        if not file_exists:
            writer.writerow(headers)
        
        # 写入本次实验数据
        writer.writerow(values)