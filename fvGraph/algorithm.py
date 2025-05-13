import numpy as np
import torch
from fvGraph.core import Module2D

def get_gradient(data:any, phi:torch.Tensor)->torch.Tensor:
    """WLSQによるノード上のスカラー場の勾配を出力。

    Args:
        data (any): データ構造
        phi (torch.Tensor): ノード上スカラー場。shapeは(N, )でNはノード数。
    Returns:
        torch.Tensor: ノード上のスカラー場の勾配。shapeは(N, dim)。
    """
    node_leader, node_follower = data.edge_node
    nodepos_leader = data.node[node_leader,]
    nodepos_follower = data.node[node_follower,]
    phi_leader = phi[node_leader,]
    phi_follower = phi[node_follower,]

    relvec = nodepos_follower - nodepos_leader
    weight = 1./torch.linalg.norm(relvec, dim = 1)
    relphi = phi_follower - phi_leader

    node_num = len(data.node)
    mat11 = torch.zeros(node_num, dtype = torch.float, device = weight.device)
    mat11 = torch.index_add(mat11, 0, node_leader, (weight**2)*(relvec[:,0]**2))
    mat22 = torch.zeros(node_num, dtype = torch.float, device = weight.device)
    mat22 = torch.index_add(mat22, 0, node_leader, (weight**2)*(relvec[:,1]**2))
    mat12 = torch.zeros(node_num, dtype = torch.float, device = weight.device)
    mat12 = torch.index_add(mat12, 0, node_leader, (weight**2)*relvec[:,0]*relvec[:,1])

    mat1 = torch.stack((mat11, mat12), axis = 1).view(-1,2)
    mat2 = torch.stack((mat12, mat22), axis = 1).view(-1,2)
    mat = torch.stack((mat1, mat2), axis = -1)

    b1 = torch.zeros(node_num, dtype = torch.float, device = weight.device)
    b1 = torch.index_add(b1, 0, node_leader, (weight**2)*relvec[:,0]*relphi)
    b2 = torch.zeros(node_num, dtype = torch.float, device = weight.device)
    b2 = torch.index_add(b2, 0, node_leader, (weight**2)*relvec[:,1]*relphi)

    b = torch.stack((b1, b2), axis = 1)
    b = b.unsqueeze(2)
    mat_inv = torch.linalg.pinv(mat)
    grad_phi = torch.matmul(mat_inv, b)[:,:,0]

    return grad_phi

def get_scalar_cell(data:any, phi:torch.Tensor)->torch.Tensor:
    """ノード上定義のスカラー場からセル中心のスカラー値を出力

    Args:
        data (any): データ構造
        phi (torch.Tensor): スカラー場
    Returns:
        torch.Tensor: セル中心スカラー場
    """
    pos = data.cell_pos
    cell_node = data.cell_node
    cell_num = len(pos)
    node_num = cell_node.shape[1]

    cell_indice = torch.arange(cell_num, dtype=torch.long, device = phi.device).repeat_interleave(node_num)
    cell_node = torch.flatten(cell_node)

    relvec = pos[cell_indice,] - data.node[cell_node,]

    phiv = phi[cell_node,]
    grad_phi = get_gradient(data, phi)[cell_node,]

    phic = torch.zeros(cell_num, dtype = torch.float, device = phi.device)
    phic = torch.index_add(phic, 0, cell_indice, phiv + torch.sum(grad_phi*relvec, dim = 1))/node_num

    return phic

def get_vector_cell(data:any, phi:torch.Tensor)->torch.Tensor:
    """ノード上のベクトル場からセル中心のベクトル場を出力

    Args:
        data (any): データ構造
        phi (torch.Tensor): ノード上ベクトル場
    Returns:
        torch.Tensor: セル上ベクトル場
    """
    dim = phi.shape[1]
    phic = torch.stack([get_scalar_cell(data, phi[:,i]) for i in range(dim)], dim = 1)
    return phic

def get_gradient_cell(data:any, grad_phi:torch.Tensor)->torch.Tensor:
    """ノード上の勾配からセル中心の勾配を出力

    Args:
        data (any): データ構造
        grad_phi (torch.Tensor):ノード上勾配。shapeは(N, 2)
    Returns:
        torch.Tensor: セル中心の勾配。shapeは(C, 2)
    Note:
        * 下記参考文献に従い、単なる平均値計算
        <ref> Learning to solve PDEs with finite volume-informed neural networks in a data-free approach
    """
    cell_node = data.cell_node
    cell_num, node_num = cell_node.shape

    cell_indice = torch.arange(cell_num, dtype=torch.long, device = grad_phi.device).repeat_interleave(node_num)
    cell_node = torch.flatten(cell_node)

    gphi = grad_phi[cell_node,]
    gphic = []
    for i in range(gphi.shape[1]):
        gp = torch.zeros(cell_num, dtype = torch.float, device = grad_phi.device)
        gp = torch.index_add(gp, 0, cell_indice, gphi[:,i])/node_num
        gphic.append(gp)
    gphic = torch.stack(gphic, axis = 1)

    return gphic

def get_gradient_facet(data:any, grad_phi:torch.Tensor)->torch.Tensor:
    """ノード上の勾配からファセット中心の勾配を出力

    Args:
        data (fvgnn2d): fvgnnデータ
        grad_phi (torch.Tensor):ノード上勾配。shapeは(N, 2)
    Returns:
        torch.Tensor: ファセット中心の勾配。shapeは(E, 2)
    Note:
        * 下記参考文献に従い、単なる平均値計算
        <ref> Learning to solve PDEs with finite volume-informed neural networks in a data-free approach
    """
    adjmat = data.edge_node
    grad_phi_facet = 0.5*(grad_phi[adjmat[0],] + grad_phi[adjmat[1],])

    return grad_phi_facet