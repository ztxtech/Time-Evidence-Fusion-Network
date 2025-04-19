import json
from pathlib import Path


def mod(data):
    if isinstance(data, dict):
        data["kernel_activation"] = None
        data["use_T_model"] = True
        data["use_C_model"] = True
        data["fusion_method"] = 'add'
        data["use_probabilistic_layer"] = False
        if 'noise' not in data:
            data["noise"] = False
    return data


def process_json_files(folder_path):
    folder = Path(folder_path)
    # 查找所有的 JSON 文件
    for json_file in folder.rglob('*.json'):
        try:
            # 读取 JSON 文件
            with json_file.open('r', encoding='utf-8') as file:
                data = json.load(file)
            # 编辑 JSON 数据
            edited_data = mod(data)
            # 保存编辑后的 JSON 数据
            with json_file.open('w', encoding='utf-8') as file:
                json.dump(edited_data, file, ensure_ascii=False, indent=4)
            print(f"Processed {json_file}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")


if __name__ == "__main__":
    # 请将这里替换为你实际的文件夹路径
    folder_path = './configs'
    process_json_files(folder_path)
