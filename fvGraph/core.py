import meshu
import sys
import numpy as np


class Module2D:
    """2次元用FVMグラフの抽象クラス

    Attributes:
        mesh (meshu.Mesh): Meshオブジェクト。
        nodes (np.ndarray): ノード座標値。shapeは(N, dim)。
        edge_node (np.ndarray): ノード間隣接行列(COO形式)によるエッジ情報。shapeは(2, E)。
        edge_pos (np.ndarray): エッジ中心座標値。エッジ番号順に定義。shapeは(E, dim)。
        edge_relvec (np.ndarray): エッジ相対ベクトル。エッジ番号順に定義。shapeは(E, dim)
                                e = (i, j)なるエッジの場合、relvec = nodes[j] - nodes[i]
        edge_size (np.ndarray): エッジのユークリッド距離。shapeは(E, )
        cell_node (np.ndarray): ノード番号で定義されたセル情報。shapeは(C, n)でnはセルを構成するノードの数。
    Notes:
        * 同じ要素形状で構成されていることを仮定。
    """
    def __init__(self, filename:str, double_direction:bool = True)->None:
        """__init__

        Args:
            filename (str): mshファイル名
            double_direction (bool): エッジ(i, j), (j, i)のように、同じ線かつ反対向きのエッジを重複とみなさず定義するか否か。
        """
        self.mesh = meshu.Mesh(filename, 2)
        meshu.algorithm.renumbering_node(self.mesh)
        elements2d = meshu.utils.get_elements(self.mesh, 2)
        elements1d = meshu.utils.get_elements(self.mesh, 1)


        self.node = self.mesh.Nodes
        self.edge_node = meshu.algorithm.get_adjacency_matrix(self.mesh, double_direction = double_direction)
        self.edge_pos = (self.node[self.edge_node[0],] + self.node[self.edge_node[1],])/2.
        self.edge_relvec = (self.node[self.edge_node[1],] - self.node[self.edge_node[0],])
        self.edge_size = np.linalg.norm(self.edge_relvec, axis = 1)        
        self.cells_node = np.stack([element["node_tag"] for element in elements2d])
        self.phys_tag_node = meshu.utils.get_phystag_node(self.mesh)

#         self.phys_tag_node = -np.ones(len(self.nodes))
#         self.phys_tag_edge = -np.ones(self.edges.shape[1])
#         for element1d in elements1d:
#             for n in element1d["node_tag"]:
#                 self.phys_tag_node[n] = element1d["phys_tag"]
            
#             indice = np.where(
#                 (element1d["node_tag"][0] == self.edges[0])*(element1d["node_tag"][1] == self.edges[1])
#                 + (element1d["node_tag"][1] == self.edges[0])*(element1d["node_tag"][0] == self.edges[1])
#                 )[0]
#             for idx in indice:
#                 self.phys_tag_edge[idx] = element1d["phys_tag"]
        
#         x_st = self.nodes[self.edges[0],0]
#         y_st = self.nodes[self.edges[0],1]
#         x_fn = self.nodes[self.edges[1],0]
#         y_fn = self.nodes[self.edges[1],1]
#         self.edge_normal = np.stack((y_fn - y_st, x_st - x_fn), axis = 1)/(self.edge_size.reshape((-1,1)))


#         self.cells_edge = []
#         for cell_node in self.cells_node:
#             ce = []
#             node_ex = np.concatenate((cell_node, np.array([cell_node[0]])))
#             for n_st, n_fn in zip(node_ex[:-1], node_ex[1:]):
#                 idx = np.where((self.edges[0] == n_st)*(self.edges[1] == n_fn))[0]
#                 if len(idx) == 0:
#                     idx = np.where((self.edges[1] == n_st)*(self.edges[0] == n_fn))[0][0]
#                     ce.append(-idx)
#                 else:
#                     ce.append(idx[0])
#             self.cells_edge.append(np.array(ce))
#         self.cells_edge = np.stack(self.cells_edge)

#         self.cell_size = np.array([meshu.geom.get_volume(element2d, self.mesh) for element2d in elements2d])
#         self.cell_pos = np.stack([np.mean(self.nodes[cn], axis = 0) for cn in self.cells_node])