# 修改后的递归版本
import json
from pathlib import Path

from utils.tools import get_all_json_paths

if __name__ == "__main__":

    alb_args = {
        'use_norm': [False],
        'use_T_model': [False],
        'use_C_model': [False],
        'fusion_method': ['concat'],
        'kernel_activation': ['relu', 'gelu', 'swish', 'mish', 'linear', 'elu', 'tanh'],
        'use_probabilistic_layer': [True]
    }

    configs_path = Path('./configs/comparision').resolve()
    ablation_path = Path('./configs/ablation').resolve()
    jsons = get_all_json_paths(configs_path, recursive=True)

    for json_path in jsons:
        with open(json_path, 'r') as f:
            original_config = json.load(f)

        for key, values in alb_args.items():
            for v in values:
                modified_config = original_config.copy()
                modified_config[key] = v

                # 在构造new_path时添加以下逻辑：
                original_path = Path(json_path)
                relative_parent = original_path.parent.relative_to(configs_path)
                new_parent_dir = Path(ablation_path) / relative_parent

                new_filename = f"{original_path.stem}_{key}_{v}.json"
                new_path = new_parent_dir / new_filename

                # 在保存前确保目标目录存在：
                new_parent_dir.mkdir(parents=True, exist_ok=True)

                # 完整的new_path赋值部分：
                new_parent_dir = (Path(ablation_path) / original_path.relative_to(configs_path).parent)
                new_parent_dir.mkdir(parents=True, exist_ok=True)
                new_path = new_parent_dir / f"{original_path.stem}_{key}_{v}.json"

                # 5. 保存修改后的配置到新文件
                with open(new_path, 'w') as f:
                    json.dump(modified_config, f, indent=4)
