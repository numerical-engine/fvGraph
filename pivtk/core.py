import numpy as np
from copy import deepcopy

class version2:
    """VTK version2を管理する抽象クラス

    geom.pyの各クラスに継承される

    Attributes:
        geom_type (str): ジオメトリタイプ。"UNSTRUCTURED_GRID"など文字列はVTKフォーマットに従う。
        point_data (list[dict]): ポイントデータのリスト。各要素はdictで、key及びvaluesは以下の通り。
            "name" (str): ポイントデータ名
            "values" (np.ndarray): 数値データ。スカラーの場合shapeは(N, )、スカラーの場合は(N, D)。
            "type" (str): "scalar"もしくは"vector"
        cell_data (list[str]): セルデータのリスト。各要素はdictで、key及びvaluesは同上。
    """
    geom_type = None
    def __init__(self, point_data:list[dict] = [], cell_data:list[dict] = [])->None:
        self.point_data = deepcopy(point_data)
        self.cell_data = deepcopy(cell_data)
    
    @property
    def dim(self)->int:
        """次元数を出力
        """
        raise NotImplementedError
    @property
    def num_points(self)->int:
        """節点の数を出力
        """
        raise NotImplementedError
    @property
    def num_cells(self)->int:
        """要素の数を出力
        """
        raise NotImplementedError
    
    def add_pointdata(self, name:str, values:np.ndarray, theresold:float = 1e-10)->None:
        """ポイントデータを追加

        Args:
            name (str): ポイントデータの名前
            values (np.ndarray): 数値データ
        """
        assert len(values) == self.num_points
        v = deepcopy(values)
        v[np.abs(v) < theresold] = 0.
        if len(values.shape) == 1:
            self.point_data.append({"name" : name, "values" : v, "type" : "scalar"})
        else:
            self.point_data.append({"name" : name, "values" : v, "type" : "vector"})
    
    def add_celldata(self, name:str, values:np.ndarray)->None:
        """セルデータを追加

        Args:
            name (str): セルデータの名前
            values (np.ndarray): 数値データ
        """
        assert len(values) == self.num_cells
        if len(values.shape) == 1:
            self.cell_data.append({"name" : name, "values" : values, "type" : "scalar"})
        else:
            self.cell_data.append({"name" : name, "values" : values, "type" : "vector"})
    
    def write_dataset(self, filename:str)->None:
        raise NotImplementedError
    
    def write_scalar(self, name : str, values : np.ndarray, filename : str)->None:
        with open(filename, "a") as file:
            file.write("SCALARS {} float 1\n".format(name))
            file.write("LOOKUP_TABLE default\n")
            for v in values:
                file.write("{}\n".format(v))
    
    def np2str(self, L : np.ndarray)->str:
        s = str(L[0])
        for l in L[1:]:
            s += " " + str(l)
        
        return s + "\n"
    
    def write_vector(self, name : str, values : np.ndarray, filename : str)->None:
        _values = np.concatenate((values, np.zeros((len(values), 1))), axis = 1) if self.dim == 2 else values
        
        with open(filename, "a") as file:
            file.write("VECTORS {} float\n".format(name))
            for v in _values:
                file.write(self.np2str(v))

    def write_pointdata(self, filename : str)->None:
        if not self.point_data: return
        with open(filename, "a") as file:
            file.write("POINT_DATA {}\n".format(self.num_points))
        
        for point_data in self.point_data:
            if point_data["type"] == "scalar":
                self.write_scalar(point_data["name"], point_data["values"], filename)
            else:
                self.write_vector(point_data["name"], point_data["values"], filename)

    def write_celldata(self, filename : str)->None:
        if not self.cell_data: return
        with open(filename, "a") as file:
            file.write("CELL_DATA {}\n".format(self.num_cells))

        for cell_data in self.cell_data:
            if cell_data["type"] == "scalar":
                self.write_scalar(cell_data["name"], cell_data["values"], filename)
            else:
                self.write_vector(cell_data["name"], cell_data["values"], filename)

    def write(self, filename : str)->None:
        """VTKファイルを出力

        Args:
            filename (str): ファイル名
        """
        with open(filename, "w") as file:
            file.write("# vtk DataFile Version 2.0\n")
            file.write("VTKio\n")
            file.write("ASCII\n")
            file.write("DATASET {}\n".format(self.geom_type))
        self.write_dataset(filename)
        self.write_pointdata(filename)
        self.write_celldata(filename)