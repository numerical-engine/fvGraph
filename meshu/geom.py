import numpy as np
from meshu import config
from meshu.core import Mesh
import sys

def get_centroid(element:dict, mesh:Mesh)->np.ndarray:
    """要素の重心を出力

    Args:
        element (dict): 要素情報
        mesh (Mesh): Meshオブジェクト
    Returns:
        np.ndarray: 重心座標。shapeは(D, )。
    """
    nodes = mesh.Nodes[element["node_tag"],]
    centroid = np.mean(nodes, axis = 0)

    return centroid


def get_volume(element:dict, mesh:Mesh)->float:
    """要素の体積(2次元の場合は面積)を出力

    Args:
        element (dict): 要素情報
        mesh (Mesh): Meshオブジェクト
    Returns:
        float: 体積(もしくは面積)
    Note:
        * 2次元の場合、要素は自己交差無しの多角形であること、並びに節点が反時計回りの順に定義されていることを仮定
    """
    if mesh.dim == 2:
        nodes = mesh.Nodes[element["node_tag"],:]
        nodes_ex = np.concatenate((nodes, nodes[0].reshape((1,2))), axis = 0)
        V = 0.
        for i in range(len(nodes)):
            V += (nodes_ex[i,0] - nodes_ex[i+1,0])*(nodes_ex[i,1] + nodes_ex[i+1,1])
        V *= 0.5
        return V
    else:
        raise NotImplementedError


def get_facet_normal(element:dict, mesh:Mesh)->tuple[np.ndarray]:
    """要素にある各ファセットの外向き単位法線ベクトルを出力

    Args:
        element (dict): 要素情報
        mesh (Mesh): Meshオブジェクト
    Returns:
        tuple[np.ndarray]: 外向き単位法線ベクトル
    Note:
        * 2次元の場合、要素は自己交差無しの多角形であること、並びに節点が反時計回りの順に定義されていることを仮定
    """
    if mesh.dim == 2:
        nodes = mesh.Nodes[element["node_tag"],:]
        nodes_ex = np.concatenate((nodes, nodes[0].reshape((1,2))), axis = 0)
        facet_normal = []
        for i in range(len(nodes)):
            x1, y1 = nodes_ex[i]; x2, y2 = nodes_ex[i+1]
            l = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            n = np.array([y2-y1, x1-x2])/l
            facet_normal.append(n)
        
        return tuple(facet_normal)
    else:
        raise NotImplementedError