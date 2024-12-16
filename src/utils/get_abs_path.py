import os
from src.config.default_config import default_config
# get the absolute path of the project

def get_abs_path():
    if default_config["use_default_path"]:
        return default_config["default_path"]
    else:
        return os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


if __name__ == "__main__":
    print(get_abs_path())
    # 列出当前目录下所有的文件
    print(os.listdir(get_abs_path()))