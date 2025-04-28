import numpy as np
from meshu import config, utils
from meshu.core import Mesh
import sys
from scipy.sparse import dok_matrix, triu

def get_adjacency_matrix(mesh:Mesh, include_selfloop:bool = False, double_direction:bool = False)->np.ndarray:
    """メッシュ構造の隣接行列(COO形式)を出力

    Args:
        mesh (Mesh): MSHオブジェクト
        include_selfroop (bool, optional): 自己ループを加えるか否か。
        double_direction (bool, optional): Trueの場合、(i,j), (j, i)両方がCOOに含む冗長な表現を出力
    Returns:
        np.ndarray: 隣接行列。shapeは(2, E)でEはエッジ数。
    Note:
        * 無向グラフを出力。
        * 隣接行列の要素値は節点タグ。
    """
    element_tags = utils.pickup_elementtag(mesh, mesh.dim)
    elements = [mesh.Elements[et] for et in element_tags]
    num_node = len(mesh.Nodes)
    
    A = dok_matrix((num_node, num_node), dtype = int)
    for element in elements:
        node_tag = element["node_tag"]
        for i in range(len(node_tag)-1):
            A[node_tag[i], node_tag[i+1]] = 1; A[node_tag[i+1], node_tag[i]] = 1
        A[node_tag[0], node_tag[-1]] = 1; A[node_tag[-1], node_tag[0]] = 1
    if include_selfloop:
        for i in range(num_node):
            A[i,i] = 1

    A = A.tocoo()
    if double_direction == False:
        A = triu(A)
    A = np.stack(A.coords, axis = 0)

    return A

def get_order(mesh:Mesh)->np.ndarray:
    """各ノードのオーダーを計算

    Args:
        mesh (Mesh): Meshオブジェクト。
    Returns:
        np.ndarray: オーダー。shapeは(N, )。第i要素はi番目ノードのオーダー。
    """
    node_num = len(mesh.Nodes)
    A = get_adjacency_matrix(mesh, double_direction = True)
    order = np.array([len(np.where(A[0] == i)[0]) for i in range(node_num)])
    return order

def renumbering_node(mesh:Mesh)->None:
    """Reverse Cuthill Mckeeによる節点タグの再分配

    Args:
        mesh (Mesh): 分配前Meshオブジェクト
    """
    node_num = len(mesh.Nodes)
    A = get_adjacency_matrix(mesh, double_direction = True)
    order = np.array([len(np.where(A[0] == i)[0]) for i in range(node_num)])

    new_tag = [np.argmin(order)]
    for i in range(node_num):
        adj = A[1,np.where(A[0] == new_tag[i])[0]]
        mask = [a not in new_tag for a in adj]
        adj = adj[mask]
        adj_order = np.array([order[a] for a in adj])
        arg_sort = np.argsort(adj_order)
        adj_sort = adj[arg_sort]
        new_tag += list(adj_sort)

        if len(new_tag) == node_num:
            break
    
    new_tag = np.array(new_tag)
    mesh.Nodes = mesh.Nodes[new_tag,]
    for i in range(len(mesh.Elements)):
        new_node_tag = tuple(np.where(new_tag == old_node_tag)[0][0] for old_node_tag in mesh.Elements[i]["node_tag"])
        mesh.Elements[i]["node_tag"] = new_node_tag