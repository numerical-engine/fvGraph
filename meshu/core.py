import numpy as np
import re

class Mesh:
    """mshフォーマットで定義されたメッシュに関するクラス

    Attributes:
        dim (int): 次元
        PhysicalGroups (list[dict]) PhysicalGroupのリスト。辞書型のkeyは以下の通り
            * dim (int): PhysicalGroupの次元
            * name (int): 名前
        Nodes (np.ndarray): 全節点の座標値。shapeは(N, 3)でNは節点数。
        Elements (list[dict]): 要素のリスト。辞書型のkeyは以下の通り。
            * type (int): 要素タイプ。gmshマニュアル参照。
            * phys_tag (int): PhysicalGroupのタグ (ゼロ始まり)。
            * node_tag (tuple[int]): 要素を構成する節点のタグ (ゼロ始まり)。
    
    Note:
        * ゼロから始まるphys_tagはPhysicalGroupのインデックス番号と対応する。phys_tag == iの場合、その要素のPhysicalGroupはPhysicalGroups[i]。
        * ゼロから始まるnode_tagはNodesのインデックス番号と対応する。node_tag == iの場合、その節点はNodes[i]。
    """
    def __init__(self, filename:str, dim:int)->None:
        assert 1 <= dim <= 3
        self.dim = dim

        self.PhysicalGroups = []
        self.Nodes = []
        self.Elements = []

        with open(filename, "r") as file:
            lines = file.readlines()
        current_index = 0
        while True:
            if lines[current_index][:-1] == "$EndMeshFormat":
                current_index += 1
                break
            current_index += 1
        
        while True:
            if lines[current_index][:-1] == "$PhysicalNames":
                current_index += 1
                current_index = self.read_PhysicalGroups(lines, current_index)
            elif lines[current_index][:-1] == "$Nodes":
                current_index += 1
                current_index = self.read_Nodes(lines, current_index)
            elif lines[current_index][:-1] == "$Elements":
                current_index += 1
                current_index = self.read_Elements(lines, current_index)
            else:
                NotImplementedError

            if current_index == len(lines):
                break
    
    def read_PhysicalGroups(self, lines:list[str], current_index:int)->int:
        phys_num = int(lines[current_index])
        current_index += 1
        
        for idx in range(phys_num):
            phys_info = re.split("[ \t]", lines[current_index][:-1])
            current_index += 1

            dim, tag, name = int(phys_info[0]), int(phys_info[1]), phys_info[2][1:-1]
            assert tag == idx + 1, "Tag of PhysicalGroups should be dense"
            assert dim <= self.dim
            self.PhysicalGroups.append({"dim":dim, "name":name})
        
        assert lines[current_index][:-1] == "$EndPhysicalNames"
        return current_index + 1
    
    def read_Nodes(self, lines, current_index):
        node_num = int(lines[current_index])
        current_index += 1
        for idx in range(node_num):
            node_info = re.split("[ \t]", lines[current_index][:-1])
            current_index += 1

            tag = int(node_info[0])
            assert tag == idx + 1, "Tag of Nodes should be dense"
            
            x, y, z = tuple(float(n) for n in node_info[1:])
            self.Nodes.append([x, y, z])
        
        self.Nodes = np.stack(self.Nodes)
        assert np.all(np.isclose(self.Nodes[:,self.dim:], 0.))
        
        self.Nodes = self.Nodes[:,:self.dim]
        assert self.Nodes.shape == (node_num, self.dim), f"{self.Nodes.shape}"

        assert lines[current_index][:-1] == "$EndNodes"

        return current_index + 1
    
    def read_Elements(self, lines, current_index):
        element_num = int(lines[current_index])
        current_index += 1

        for idx in range(element_num):
            element_info = [int(e) for e in re.split("[ \t]", lines[current_index][:-1])]
            current_index += 1

            assert element_info[0] == idx + 1, "Tag of Element should be dense"
            assert element_info[2] == 2
            e_type = element_info[1]
            phys_tag = element_info[3] - 1 #start from zero
            node_tag = tuple(e - 1 for e in element_info[5:]) #start from zero

            self.Elements.append({"type":e_type, "phys_tag":phys_tag, "node_tag":node_tag})
        
        assert lines[current_index][:-1] == "$EndElements"

        return current_index + 1
    
    def write(self, filename:str)->None:
        """mshファイルの書き出し

        Args:
            filename (str): ファイル名

        Note:
            * ファイルフォーマットはversion 2.2
        """
        with open(filename, "w") as file:
            file.write("$MeshFormat\n")
            file.write("2.2 0 8\n")
            file.write("$EndMeshFormat\n")
            
            phys_num = len(self.PhysicalGroups)
            if phys_num > 0:
                file.write("$PhysicalNames\n")
                file.write(f"{phys_num}\n")
                for idx, phys_G in enumerate(self.PhysicalGroups):
                    dim = phys_G["dim"]
                    name = phys_G["name"]
                    file.write(f"{dim} {idx+1} \"{name}\"\n")
                file.write("$EndPhysicalNames\n")
            
            file.write("$Nodes\n")
            file.write(f"{len(self.Nodes)}\n")
            for idx, node in enumerate(self.Nodes):
                node_ex = np.concatenate((node, np.zeros(3-len(node))))
                
                file.write(f"{idx+1} {node_ex[0]} {node_ex[1]} {node_ex[2]}\n")
            file.write("$EndNodes\n")
            
            file.write("$Elements\n")
            file.write(f"{len(self.Elements)}\n")
            for idx, element in enumerate(self.Elements):
                e_type = element["type"]
                phys_tag = element["phys_tag"]
                node_tag = [n+1 for n in element["node_tag"]]
                node_tag_str = ""
                for n in node_tag:
                    node_tag_str += f"{int(n)} "
                node_tag_str = node_tag_str[:-1]+"\n"
                file.write(f"{idx+1} {e_type} 2 {phys_tag} 1 {node_tag_str}")
            file.write("$EndElements\n")