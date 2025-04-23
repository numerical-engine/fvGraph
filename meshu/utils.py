import numpy as np
from meshu import config
from meshu.core import Mesh
import pivtk

def pickup_elementtag(mesh:Mesh, dim:int)->tuple[int]:
    """次元数がdimの要素タグを出力

    Args:
        mesh (Mesh): Meshオブジェクト 
        dim (int): 次元
    Returns:
        tuple[int]: 該当要素のタグのリスト (ゼロ始まり)
    """
    element_tag = []

    for tag, element in enumerate(mesh.Elements):
        e_type = element["type"]
        if e_type in config.element_types[dim]:
            element_tag.append(tag)
    return tuple(element_tag)


def get_elements(mesh:Mesh, dim:int)->tuple[dict]:
    """次元がdimの要素のタプルを出力

    Args:
        mesh (Mesh): Meshオブジェクト
        dim (int): 次元
    Returns:
        tuple[dict]: 要素情報のタプル
    """
    tags = pickup_elementtag(mesh, dim)
    elements = tuple([mesh.Elements[t] for t in tags])

    return elements


def get_physical_names(mesh:Mesh, dim:int = None)->tuple[str]:
    """次元がdimのPhysicalGroupの名前を出力

    Args:
        mesh (Mesh): Meshオブジェクト
        dim (int, optional): PhysicalGroupの次元。Noneの場合すべてのPhysicalGroupの名前を出力
    Returns:
        tuple[str]: 名前のタプル
    """
    names = []
    for phys_g in mesh.PhysicalGroups:
        if dim is None:
            names.append(phys_g["name"])
        else:
            if phys_g["dim"] == dim:
                names.append(phys_g["name"])
    
    return tuple(names)


def Graph2UnstructuredGrid(V:np.ndarray, E:np.ndarray)->pivtk.geom.unstructured_grid:
    """ノード座標値と隣接行列(COO形式)からVTKジオメトリを出力

    Args:
        V (np.ndarray): ノード座標値。
        E (np.ndarray): 隣接行列。
    Returns:
        geom.unstructured_grid: unstructured gridジオメトリ
    """
    cells = [{"type" : 3, "indice" : np.array([e_st,e_fn])} for e_st, e_fn in zip(E[0], E[1])]
    return pivtk.geom.unstructured_grid(points = V, cells = tuple(cells))