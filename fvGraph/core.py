import meshu
import sys
import os
import numpy as np



class Module2D:
    """2次元用FVMグラフの抽象クラス

    Attributes:
        mesh (meshu.Mesh): Meshオブジェクト。
        node (np.ndarray): ノード座標値。shapeは(N, dim)。
        edge_node (np.ndarray): ノード間隣接行列(COO形式)によるエッジ情報。shapeは(2, E)。
        edge_pos (np.ndarray): エッジ中心座標値。エッジ番号順に定義。shapeは(E, dim)。
        edge_relvec (np.ndarray): エッジ相対ベクトル。エッジ番号順に定義。shapeは(E, dim)
                                e = (i, j)なるエッジの場合、relvec = nodes[j] - nodes[i]
        edge_size (np.ndarray): エッジのユークリッド距離。shapeは(E, )
        cell_node (np.ndarray): ノード番号で定義されたセル情報。shapeは(C, n)でnはセルを構成するノードの数。
        cell_edge (np.ndarray): エッジ番号で定義されたセル情報。shapeは(C, e)でeはセルを構成するエッジの数。
        cell_size (np.ndarray): セルの面積。shapeは(C, )。
        cell_pos (np.ndarray): セルの中心座標。shapeは(C, dim)。
        interface_cell (np.ndarray): セル中心の隣接行列。shapeは(2, Ce)でCeはセル界面の数。境界のエッジが含まれない分、Ce < E。
        interface_edge (np.ndarray): セル間の界面のエッジ番号。shapeは(Ce, )
        phys_tag_node (np.ndarrat): 各ノードの境界物理タグ。shapeは(N, )。境界に居ないノードのタグは-1。
        phys_tag_edge (np.ndarrat): 各エッジの境界物理タグ。shapeは(E, )。境界に居ないエッジのタグは-1。
        phys_tag_cell (np.ndarrat): 各セルの物理タグ。shapeは(C, )。
    Notes:
        * 同じ要素形状で構成されていることを仮定。
        * double direction。
    """
    def __init__(self, filename:str = None, load_dir:str = None)->None:
        """__init__

        Args:
            filename (str): mshファイル名。Noneの場合、既出データファイルから読み込み。
            load_dir (str): loadディレクトリ。
        """
        if filename is None:
            self.load(load_dir)
        else:
            self.mesh = meshu.Mesh(filename, 2)
            meshu.algorithm.renumbering_node(self.mesh)

            self.node = self.mesh.Nodes
            self.edge_node = meshu.algorithm.get_adjacency_matrix(self.mesh, double_direction = True)
            self.edge_pos = (self.node[self.edge_node[0],] + self.node[self.edge_node[1],])/2.
            self.edge_relvec = (self.node[self.edge_node[1],] - self.node[self.edge_node[0],])
            self.edge_size = np.linalg.norm(self.edge_relvec, axis = 1)        
            self.cell_node = np.stack([element["node_tag"] for element in meshu.utils.get_elements(self.mesh, 2)])
            self.phys_tag_node = meshu.utils.get_phystag_node(self.mesh)
            self.phys_tag_edge = meshu.utils.get_phystag_COO(self.mesh, self.edge_node, except_val = -1)
            self.phys_tag_cell = np.stack([element["phys_tag"] for element in meshu.utils.get_elements(self.mesh, 2)])
            self.edge_normal = np.stack([meshu.geom.get_facet_normal_between_nodes(self.mesh, i, j)
                                        for i, j in zip(self.edge_node[0], self.edge_node[1])])
            self.cell_edge = np.stack([np.array(meshu.utils.get_element_edge_list(element, self.edge_node))
                                    for element in meshu.utils.get_elements(self.mesh, 2)])
            self.cell_size = np.array([meshu.geom.get_volume(element, self.mesh) for element in meshu.utils.get_elements(self.mesh, 2)])
            self.cell_pos = np.stack([np.mean(self.node[cn], axis = 0) for cn in self.cell_node])

            self.interface_cell = []
            self.interface_edge = []
            for idx, cell_node_i in enumerate(self.cell_node):
                for jdx, cell_node_j in enumerate(self.cell_node):
                    if jdx <= idx: continue
                    itst = np.intersect1d(cell_node_i, cell_node_j)
                    if len(itst) == 2:
                        edge_i = np.where((self.edge_node[0] == itst[0])*(self.edge_node[1] == itst[1]))[0][0]
                        edge_j = np.where((self.edge_node[0] == itst[1])*(self.edge_node[1] == itst[0]))[0][0]

                        self.interface_cell += [np.array([idx, jdx]), np.array([jdx, idx])]
                        self.interface_edge += [edge_i, edge_j]
            self.interface_cell = np.stack(self.interface_cell, axis = 1)
            self.interface_edge = np.array(self.interface_edge)
    

    def save(self, save_dir:str)->None:
        """データ構造を保存

        Args:
            save_dir (str): 保存先
        Note:
            * 各属性がnp.ndarrayでないとエラー。
        """
        assert isinstance(self.node, np.ndarray)
        os.makedirs(save_dir, exist_ok=True)
        self.mesh.write(f"{save_dir}/mesh.msh")
        np.save(f"{save_dir}/node.npy", self.node)
        np.save(f"{save_dir}/edge_node.npy", self.edge_node)
        np.save(f"{save_dir}/edge_pos.npy", self.edge_pos)
        np.save(f"{save_dir}/edge_relvec.npy", self.edge_relvec)
        np.save(f"{save_dir}/edge_size.npy", self.edge_size)
        np.save(f"{save_dir}/cell_node.npy", self.cell_node)
        np.save(f"{save_dir}/phys_tag_node.npy", self.phys_tag_node)
        np.save(f"{save_dir}/phys_tag_edge.npy", self.phys_tag_edge)
        np.save(f"{save_dir}/phys_tag_cell.npy", self.phys_tag_cell)
        np.save(f"{save_dir}/edge_normal.npy", self.edge_normal)
        np.save(f"{save_dir}/cell_edge.npy", self.cell_edge)
        np.save(f"{save_dir}/cell_size.npy", self.cell_size)
        np.save(f"{save_dir}/cell_pos.npy", self.cell_pos)
        np.save(f"{save_dir}/interface_cell.npy", self.interface_cell)
        np.save(f"{save_dir}/interface_edge.npy", self.interface_edge)
    
    def load(self, load_dir:str)->None:
        self.mesh = meshu.Mesh(f"{load_dir}/mesh.msh", 2)
        self.node = np.load(f"{load_dir}/node.npy")
        self.edge_node = np.load(f"{load_dir}/edge_node.npy")
        self.edge_pos = np.load(f"{load_dir}/edge_pos.npy")
        self.edge_relvec = np.load(f"{load_dir}/edge_relvec.npy")
        self.edge_size = np.load(f"{load_dir}/edge_size.npy")
        self.cell_node = np.load(f"{load_dir}/cell_node.npy")
        self.phys_tag_node = np.load(f"{load_dir}/phys_tag_node.npy")
        self.phys_tag_edge = np.load(f"{load_dir}/phys_tag_edge.npy")
        self.phys_tag_cell = np.load(f"{load_dir}/phys_tag_cell.npy")
        self.edge_normal = np.load(f"{load_dir}/edge_normal.npy")
        self.cell_edge = np.load(f"{load_dir}/cell_edge.npy")
        self.cell_size = np.load(f"{load_dir}/cell_size.npy")
        self.cell_pos = np.load(f"{load_dir}/cell_pos.npy")
        self.interface_cell = np.load(f"{load_dir}/interface_cell.npy")
        self.interface_edge = np.load(f"{load_dir}/interface_edge.npy")