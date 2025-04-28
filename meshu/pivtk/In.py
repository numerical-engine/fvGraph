import numpy as np
from pivtk import geom
import re
import sys

def read(filename:str)->any:
    """VTKファイルを読み込み、ジオメトリクラスを返す

    Args:
        filename (str): ファイル名
    Returns:
        any: ジオメトリクラス
    """
    with open(filename, "r") as file:
        lines = file.readlines()[3:]
    
    data_type = lines[0].split(" ")[-1][:-1]
    if data_type == "UNSTRUCTURED_GRID":
        return read_UnstructuredGrid(lines[1:])
    else:
        raise NotImplementedError


def read_data(lines:list[str], current_idx:int, point_num:int, cell_num:int)->tuple[list[dict]]:
    line_num = len(lines)
    point_data = []
    cell_data = []
    while current_idx < line_num-1:
        if lines[current_idx][:10] == "POINT_DATA":
            while current_idx < line_num-1:
                current_idx += 1
                data_type, name = re.split("[ \t]", lines[current_idx])[:2]
                if data_type == "SCALARS":
                    current_idx += 2
                    val = np.array([float(lines[current_idx+i]) for i in range(point_num)])
                    point_data.append({"name":name, "type":"scalar", "values":val})
                    current_idx += point_num-1

                elif data_type == "VECTORS":
                    current_idx += 1
                    val = np.stack([np.array([float(v) for v in re.split("[ \t]", lines[current_idx+i])]) for i in range(point_num)], axis = 0)
                    point_data.append({"name":name, "type":"vector", "values":val})
                    current_idx += point_num - 1
                else:
                    break
        else:
            while current_idx < line_num-1:
                current_idx += 1
                data_type, name = re.split("[ \t]", lines[current_idx])[:2]
                if data_type == "SCALARS":
                    current_idx += 2
                    val = np.array([float(lines[current_idx+i]) for i in range(cell_num)])
                    cell_data.append({"name":name, "type":"scalar", "values":val})
                    current_idx += point_num-1

                elif data_type == "VECTORS":
                    current_idx += 1
                    val = np.stack([np.array([float(v) for v in re.split("[ \t]", lines[current_idx+i])]) for i in range(cell_num)], axis = 0)
                    cell_data.append({"name":name, "type":"vector", "values":val})
                    current_idx += point_num - 1
                else:
                    break
    
    return point_data, cell_data


def read_UnstructuredGrid(lines:list[str])->geom.unstructured_grid:
    current_idx = 0
    point_num = int(lines[current_idx].split(" ")[1])
    current_idx += 1

    points = []
    for _ in range(point_num):
        points.append(np.array([float(s) for s in lines[current_idx].split(" ")]))
        current_idx += 1
    points = np.stack(points)

    cell_num = int(lines[current_idx].split(" ")[1])
    current_idx += 1

    cells = []
    for _ in range(cell_num):
        indice = np.array([int(l) for l in lines[current_idx].split(" ")[1:]])
        cells.append({"indice":indice})
        current_idx += 1
    current_idx += 1 #skip "CELL_TYPES ** line"
    for i in range(cell_num):
        cells[i]["type"] = int(lines[current_idx])
        current_idx += 1
    
    point_data, cell_data = read_data(lines, current_idx, point_num, cell_num)

    
    return geom.unstructured_grid(points, cells, point_data, cell_data)
