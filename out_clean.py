import shutil
from pathlib import Path


def delete_unmatched_folders(base_dir: Path, reference_dir: Path):
    """删除 base_dir 中不在 reference_dir 的子目录列表中的文件夹"""
    # 获取参考目录的所有子目录名称
    reference_folders = {
        folder.name for folder in reference_dir.iterdir() if folder.is_dir()
    }

    # 遍历目标目录并删除不符合的文件夹
    for folder_path in base_dir.iterdir():
        if (folder_path.is_dir() and
                folder_path.name not in reference_folders):
            print(f"Deleting folder: {folder_path}")
            shutil.rmtree(folder_path)


if __name__ == "__main__":
    results_dir = Path("./out/results")
    checkpoints_dir = Path("./out/checkpoints")
    test_results_dir = Path("./out/test_results")

    # 检查参考目录是否存在
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"Error: Reference directory {results_dir} does not exist")
        exit(1)

    # 执行清理操作
    delete_unmatched_folders(checkpoints_dir, results_dir)
    delete_unmatched_folders(test_results_dir, results_dir)
    log_dir = Path("./out/log")
    shutil.rmtree(log_dir)
