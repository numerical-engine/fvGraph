import numpy as np
from pivtk.core import version2

class structured_points(version2):
    """Object for STRUCTUREDMESH

    Args:
        num_grids (tuple[int]): Number of grids at each axis
        origin (tuple[float]): Original position
        spacing (tuple[float]): Growth rate at each axis
    """
    geom_type = "STRUCTURED_POINTS"
    def __init__(self, num_grids:tuple[int], origin:tuple[float] = None, spacing:tuple[float] = None, point_data:list[dict] = [], cell_data:list[dict] = [])->None:
        super().__init__(point_data, cell_data)
        self.num_grids = num_grids
        self.origin = (0.,)*self.dim if origin is None else origin
        self.spacing = (1., )*self.dim if spacing is None else spacing
    
    @property
    def dim(self)->int: return len(self.num_grids)
    @property
    def num_points(self)->int:
        n = 1
        for g in self.num_grids: n *= g
        return n
    @property
    def num_cells(self)->int:
        n = 1
        for g in self.num_grids: n *= (g - 1)
        return n
    
    def add_pointdata(self, name : str, values : np.ndarray)->None:
        assert values.shape[:self.dim] == self.num_grids
        if len(values.shape) == self.dim:
            self.point_data.append({"name" : name, "values" : values, "type" : "scalar"})
        else:
            self.point_data.append({"name" : name, "values" : values, "type" : "vector"})
    
    def add_celldata(self, name : str, values : np.ndarray)->None:
        assert values.shape[:self.dim] == tuple((ng - 1 for ng in self.num_grids))
        if len(values.shape) == self.dim:
            self.cell_data.append({"name" : name, "values" : values, "type" : "scalar"})
        else:
            self.cell_data.append({"name" : name, "values" : values, "type" : "vector"})

    
    def write_scalar(self, name : str, values : np.ndarray, filename : str)->None:
        _values = (values.T).flatten()
        with open(filename, "a") as file:
            file.write("SCALARS {} float 1\n".format(name))
            file.write("LOOKUP_TABLE default\n")
            for v in _values:
                file.write("{}\n".format(v))
    
    def write_vector(self, name : str, values : np.ndarray, filename : str)->None:
        if self.dim == 2:
            _values = (values.transpose((1, 0, 2))).reshape((-1, 2))
            _values = np.concatenate(
                (_values, np.zeros((len(_values), 1))),
                axis = 1
                )
        else:
            _values = (values.transpose((2, 1, 0, 3))).reshape((-1, 3))
        with open(filename, "a") as file:
            file.write("VECTORS {} float\n".format(name))
            for v in _values:
                file.write(self.np2str(v))
    
    def write_dataset(self, filename : str)->None:
        if self.dim == 2:
            num_grids = (self.num_grids[0], self.num_grids[1], 1)
            origin = (self.origin[0], self.origin[1], 0.)
            spacing = (self.spacing[0], self.spacing[1], 0.)
        else:
            num_grids = self.num_grids
            origin = self.origin
            spacing = self.spacing
        
        with open(filename, "a") as file:
            file.write("DIMENSIONS {0} {1} {2}\n".format(num_grids[0], num_grids[1], num_grids[2]))
            file.write("ORIGIN {0} {1} {2}\n".format(origin[0], origin[1], origin[2]))
            file.write("SPACING {0} {1} {2}\n".format(spacing[0], spacing[1], spacing[2]))


class unstructured_grid(version2):
    """Object for UNSTRUCTUREDGRID

    Args:
        points (np.ndarray): Coordinates of each points
        cells (tuple[dict]): Information of each cells. This keys are "type" and "indice", where
            "type" (int): cell's tag
            "indice" (np.ndarray) point index array
    Attributes:
        points (np.ndarray): Coordinates of each points
        cells (tuple[dict]): Information of each cells. This keys are "type" and "indice", where
            "type" (int): cell's tag
            "indice" (np.ndarray) point index array
    """
    geom_type = "UNSTRUCTURED_GRID"
    def __init__(self, points : np.ndarray, cells : tuple, point_data:list[dict] = [], cell_data:list[dict] = [])->None:
        super().__init__(point_data, cell_data)
        self.points = points
        self.cells = cells
    
    @property
    def dim(self) -> int: return self.points.shape[-1]
    @property
    def num_points(self) -> int: return len(self.points)
    @property
    def num_cells(self) -> int: return len(self.cells)

    def write_dataset(self, filename : str)->None:
        points = np.concatenate((self.points, np.zeros((self.num_points, 1))), axis = 1) if self.dim == 2 else self.points
        with open(filename, "a") as file:
            file.write("POINTS {} float\n".format(self.num_points))
            for point in points:
                file.write(self.np2str(point))
            
            size = 0
            for cell in self.cells:
                size += 1 + len(cell["indice"])
            file.write("CELLS {0} {1}\n".format(self.num_cells, size))
            for cell in self.cells:
                s = str(len(cell["indice"])) + " " + self.np2str(cell["indice"])
                file.write(s)
            
            file.write("CELL_TYPES {}\n".format(self.num_cells))
            for cell in self.cells:
                file.write("{}\n".format(cell["type"]))

class point_cloud(unstructured_grid):
    """Object for point cloud

    Args:
        points (np.ndarray): Coordinates at each points
    """
    def __init__(self, points : np.ndarray, point_data:list[dict] = [])->None:
        cells = tuple(({"type" : 1, "indice" : np.array([i])} for i in range(len(points))))
        super().__init__(points, cells, point_data, []) #cell_dataは未定義とする
    
    def add_celldata(self, name : str, values : np.ndarray)->None:
        raise Exception("point cloud can't define cell data")