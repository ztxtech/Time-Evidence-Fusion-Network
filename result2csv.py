import json
import os

import pandas as pd

from utils.tools import get_all_json_paths

if __name__ == "__main__":
    result_path = './out/results'
    arg_jsons = get_all_json_paths(result_path, recursive=True)

    # 2. 读取所有JSON文件内容
    data_list = []
    for arg_json in arg_jsons:
        with open(arg_json, 'r') as f:
            data = json.load(f)
            data_list.append(data)

    # 3. 转换为DataFrame并导出CSV
    df = pd.DataFrame(data_list)
    df.to_csv(os.path.join(result_path, "output.csv"), index=False)
