import meshu
import sys
import numpy as np

class Module2D:
    """複数のグラフを管理するためのクラス。

    各グラフで要素番号やファセット番号、並びにノード番号が共有されている。

    Attributes:
        node_pos (np.ndarray): ノード座標値。ノード番号順に定義。shapeは(N, D)
        facets (np.ndarray): ファセット情報。ファセット番号順に定義。shapeは(E, 2)。
        cells (np.ndarray): セル情報。セル番号順に定義。shapeは(C, n)でnはセルを構成するノード数。
    Note:
        * 同じ種類のセルでジオメトリが構成されていることを仮定。
    """
    def __init__(self,
                filename:str = None,
                double_direction:bool = True,
                node_pos:np.ndarray = None,
                facets:np.ndarray = None,
                cells:np.ndarray = None,
                )->None:
        """__init__

        Args:
            filename (str, optional): mshファイル名。Noneの場合Attributesを明示的に指定。
            double_direction (bool): ファセット(i, j), (j, i)のように、同じ面かつ反対向きのファセットを重複とみなさず定義するか否か。
                                    なお、double_directionの定義はmeshuのget_adjacency_matrixと同じ。
        """
        if filename is None:
            self.node_pos = node_pos
            self.facets = facets
            self.cells = cells
        else:
            mesh = meshu.Mesh(filename, 2)
            meshu.algorithm.renumbering_node(mesh)
            self.node_pos = mesh.Nodes
            self.facets = meshu.algorithm.get_adjacency_matrix(mesh, double_direction = double_direction)
            
            elements = meshu.utils.get_elements(mesh, 2)
            self.cells = np.stack([element["node_tag"] for element in elements])