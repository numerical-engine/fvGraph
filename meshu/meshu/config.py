#####gmsh要素タイプと次元の対応
element_types = {
    0 : tuple([15]),
    1 : tuple([1]),
    2 : (2, 3),
    3 : (4, 5, 6)
}

#####gmsh要素タイプとvtk要素タイプの対応
etype_msh_vtk = {
    2:5, #triangle
    3:9, #quad
    6:13, #prism
}