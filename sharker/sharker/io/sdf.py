import mindspore as ms
from mindspore import ops, Tensor
from ..data import Graph
from .txt_array import parse_txt_array
from ..utils import coalesce

elems = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}


def parse_sdf(src: str) -> Graph:
    lines = src.split("\n")[3:]
    num_atoms, num_bonds = [int(item) for item in lines[0].split()[:2]]

    atom_block = lines[1:num_atoms+1]
    crd = parse_txt_array(atom_block, end=3)
    x = Tensor([elems[item.split()[3]] for item in atom_block])
    x = ops.one_hot(x, depth=len(elems))

    bond_block = lines[1+num_atoms:1+num_atoms+num_bonds]
    row, col = parse_txt_array(bond_block, end=2, dtype=ms.int64).t() - 1
    row, col = ops.cat([row, col], axis=0), ops.cat([col, row], axis=0)
    edge_index = ops.stack([row, col], axis=0)
    edge_attr = parse_txt_array(bond_block, start=2, end=3) - 1
    edge_attr = ops.cat([edge_attr, edge_attr], axis=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_atoms)

    return Graph(x=x, edge_index=edge_index, edge_attr=edge_attr, crd=crd)


def read_sdf(path: str) -> Graph:
    with open(path, "r") as f:
        return parse_sdf(f.read())
