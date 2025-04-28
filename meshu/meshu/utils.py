import numpy as np
from meshu import config
from meshu.core import Mesh
import pivtk
import sys

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


def get_phystag_node(mesh:Mesh)->np.ndarray:
    """境界にあるノードに対し、属している境界のPhysical Tagを出力。

    境界上でないノードには、-1のTagを与える。
    Args:
        mesh (Mesh): Meshオブジェクト。
    Returns:
        np.ndarray: 境界ノードのPhysical Tag情報。
    """
    phys_tag = -np.ones(len(mesh.Nodes))
    element1d = get_elements(mesh, mesh.dim-1)
    for e1d in element1d:
        for n in e1d["node_tag"]:
            phys_tag[n] = e1d["phys_tag"]
    
    return phys_tag

def get_edge(mesh:Mesh, i:int, j:int)->dict:
    """e = (i,j)のエッジ情報を出力

    Args:
        mesh (Mesh): Meshオブジェクト。
        i (int): 開始点ノードtag。
        j (int): 終了点ノードtag。
    Returns:
        dict: エッジ情報。(i,j)なるエッジがない場合はNoneを返す。
    """
    edges = get_elements(mesh, 1)

    for edge in edges:
        if (i == edge["node_tag"][0]) & (j == edge["node_tag"][1]):
            return edge
    return None

def get_phystag_between_nodes(mesh:Mesh, i:int, j:int, except_val:int = -1)->int:
    """エッジ(i,j)のphysical tagを出力

    Args:
        mesh (Mesh): Meshオブジェクト。
        i (int): 開始点ノードtag。
        j (int): 終了点ノードtag。
        except_val (int): (i, j)がない場合に返す値。
    Returns:
        int: physical tag。
    """
    element = get_edge(mesh, i, j)
    if element is None:
        element = get_edge(mesh, j, i)
        if element is None:
            return except_val
        else:
            return element["phys_tag"]
    else:
        return element["phys_tag"]

def get_phystag_COO(mesh:Mesh, COO:np.ndarray, except_val:int = -1)->np.ndarray:
    """COO形式で書き表されたエッジ情報に対し、各エッジのphys tagを出力。

    Args:
        mesh (Mesh): Meshオブジェクト。
        COO (np.ndarray): COO形式隣接行列。shapgeは(2, E)
        except_val (int): phys_tagがない場合のtag

    Returns:
        np.ndarray: phys tag。
    """
    phys_tag = np.array([get_phystag_between_nodes(mesh, i, j, except_val) for i, j in zip(COO[0], COO[1])])
    return phys_tag

def isin_COO(COO:np.ndarray, i:int, j:int)->bool:
    """COOに(i,j)のエッジが存在するか否かを判定

    Args:
        COO (np.ndarray): COO形式隣接行列。
        i (int): 開始ノード点。
        j (int): 終了ノード点。
    Returns:
        bool: 存在する場合True。
    """
    return np.any((COO[0] == i)*(COO[1] == j))

def get_edge_list(node_tags:np.ndarray, COO:np.ndarray)->list:
    """ノード番号の順列から成る要素に対し、要素を構成するエッジ番号のリストを出力する。

    Args:
        node_tags (np.ndarray): ノード番号順列。
        COO (np.ndarray): COO形式隣接行列。
    Returns:
        list: エッジ番号eのリスト。逆向きのエッジの場合-e。
    Note:
        * node_tagsは反時計回りの順に並ぶ。
        * エッジは反時計回りの順に並ぶ。
    """
    node_tags = np.concatenate((node_tags, np.array([node_tags])))
    edge_list = []
    for n_st, n_fn in zip(node_tags[:-1], node_tags[1:]):
        idx = np.where((COO[0] == n_st)*(COO[1] == n_fn))[0]
        if len(idx) == 0:
            idx = np.where((COO[1] == n_st)*(COO[0] == n_fn))[0]
            assert len(idx) > 0
            edge_list.append(-idx)
        else:
            edge_list.append(idx[0])
    
    return edge_list


def get_element_edge_list(element:dict, COO:np.ndarray)->list:
    """要素を構成するエッジ番号のリストを出力する。

    Args:
        element (dict): 要素情報
        COO (np.ndarray): COO形式隣接行列。
    Returns:
        list: エッジ番号のリスト。
    Note:
        * エッジは反時計回りの順に並ぶ。
    """
    node_tags = element["node_tag"]
    return get_edge_list(node_tags, COO)