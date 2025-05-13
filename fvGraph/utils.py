import sys
import numpy as np
from fvGraph.core import Module2D

def onehot_phystag(module:Module2D, category:list[list[int]], target:str)->None:
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