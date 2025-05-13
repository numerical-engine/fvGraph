import sys
import numpy as np
from fvGraph.core import Module2D
from copy import deepcopy
import torch

def onehot_phystag(module:any, category:list[list[int]], target:str)->None:
    """phystagをone-hot化

    Args:
        module (Module2D): モジュール
        category (list[list[int]]): カテゴリ値。
        例えば[[1,3,4], [2,5,6]]であれば、one-hotベクトルは2次元で、1,3,4のphystagは(1, 0)のベクトルに変換される。
        target (str): node, edge, cellのいずれのphystagを変換するか。
    Return:
        None: Moduleの属性が更新される。
    Note:
        physgtagが-1の場合、ゼロベクトルに変換される。
    """
    def onehot(phys_tag:np.ndarray)->np.ndarray:
        num_category = len(category)
        onehot_tag = []
        for pt in phys_tag:
            po = np.zeros(num_category)
            for idx, cat in enumerate(category):
                if int(pt) in cat:
                    po[idx] = 1.
                    break
            onehot_tag.append(po)
        onehot_tag = np.stack(onehot_tag)
        return onehot_tag
    
    if target == "node":
        module.phys_tag_node = onehot(module.phys_tag_node)
    elif target == "edge":
        module.phys_tag_edge = onehot(module.phys_tag_edge)
    elif target == "cell":
        module.phys_tag_cell = onehot(module.phys_tag_cell)
    else:
        raise NotImplementedError

def to_torch2d(data:Module2D, device:str = "cpu", dtype:any = torch.float)->Module2D:
    data_torch = deepcopy(data)
    data_torch.node = torch.tensor(data.node, device = device, dtype = dtype)
    data_torch.edge_node = torch.tensor(data.edge_node, device = device, dtype = torch.long)
    data_torch.edge_pos = torch.tensor(data.edge_pos, device = device, dtype = dtype)
    data_torch.edge_relvec = torch.tensor(data.edge_relvec, device = device, dtype = dtype)
    data_torch.edge_size = torch.tensor(data.edge_size, device = device, dtype = dtype)
    data_torch.cell_node = torch.tensor(data.cell_node, device = device, dtype = torch.long)
    data_torch.phys_tag_node = torch.tensor(data.phys_tag_node, device = device, dtype = dtype)
    data_torch.phys_tag_edge = torch.tensor(data.phys_tag_edge, device = device, dtype = dtype)
    data_torch.phys_tag_cell = torch.tensor(data.phys_tag_cell, device = device, dtype = dtype)
    data_torch.edge_normal = torch.tensor(data.edge_normal, device = device, dtype = dtype)
    data_torch.cell_edge = torch.tensor(data.cell_edge, device = device, dtype = torch.long)
    data_torch.cell_size = torch.tensor(data.cell_size, device = device, dtype = dtype)
    data_torch.cell_pos = torch.tensor(data.cell_pos, device = device, dtype = dtype)
    data_torch.interface_cell = torch.tensor(data.interface_cell, device = device, dtype = torch.long)
    data_torch.interface_edge = torch.tensor(data.interface_edge, device = device, dtype = torch.long)

    return data_torch

def to_numpy2d(data:Module2D)->Module2D:
    data_numpy = deepcopy(data)
    if isinstance(data.node, np.ndarray):
        return data_numpy
    else:
        data_numpy.node = data.node.cpu().numpy()
        data_numpy.edge_node = data.edge_node.cpu().numpy()
        data_numpy.edge_pos = data.edge_pos.cpu().numpy()
        data_numpy.edge_relvec = data.edge_relvec.cpu().numpy()
        data_numpy.edge_size = data.edge_size.cpu().numpy()
        data_numpy.cell_node = data.cell_node.cpu().numpy()
        data_numpy.phys_tag_node = data.phys_tag_node.cpu().numpy()
        data_numpy.phys_tag_edge = data.phys_tag_edge.cpu().numpy()
        data_numpy.phys_tag_cell = data.phys_tag_cell.cpu().numpy()
        data_numpy.edge_normal = data.edge_normal.cpu().numpy()
        data_numpy.cell_edge = data.cell_edge.cpu().numpy()
        data_numpy.cell_size = data.cell_size.cpu().numpy()
        data_numpy.cell_pos = data.cell_pos.cpu().numpy()
        data_numpy.interface_cell = data.interface_cell.cpu().numpy()
        data_numpy.interface_edge = data.interface_edge.cpu().numpy()

        return data_numpy