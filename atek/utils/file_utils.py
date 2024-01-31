import json
from typing import List


def read_txt(txt_file: str):
    with open(txt_file, "r") as f:
        lines = f.read().splitlines()
    return lines


def write_txt(lines: List[str], txt_file: str):
    with open(txt_file, "w") as f:
        f.writelines("\n".join(lines))


def read_json(json_file: str):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def write_json(data, json_file: str, indent: int = 2):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=indent)
