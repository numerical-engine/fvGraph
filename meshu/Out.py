import numpy as np
import pivtk
from meshu import core, config, utils

def getVTK(mesh:core.Mesh)->pivtk.unstructured_grid:
    """MeshオブジェクトからVTKファイルを作成

    Args:
        mesh (core.Mesh): Meshオブジェクト
    Returns:
        pivtkのunstructured gridオブジェクト
    """
    assert mesh.dim > 1

    cells = []
    element_tags = utils.pickup_elementtag(mesh, mesh.dim)
    for element_tag in element_tags:
        element = mesh.Elements[element_tag]
        cell_type = config.etype_msh_vtk[element["type"]]
        cell_indice = np.array(element["node_tag"])
        cells.append({"type" : cell_type, "indice" : cell_indice})
    
    geom = pivtk.unstructured_grid(mesh.Nodes, cells)
    return geom