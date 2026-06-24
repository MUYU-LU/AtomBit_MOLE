from typing import List

from mindspore import Tensor, ops, nn
from mindspore import ops
from ..data import Graph
from .txt_array import parse_txt_array


def parse_off(src: List[str]) -> Graph:
    # Some files may contain a bug and do not have a carriage return after OFF.
    if src[0] == "OFF":
        src = src[1:]
    else:
        src[0] = src[0][3:]

    num_nodes, num_faces = [int(item) for item in src[0].split()[:2]]
 
    crd = parse_txt_array(src[1:1+num_nodes])

    face = face_to_tri(src[1+num_nodes:1+num_nodes+num_faces])

    data = Graph(crd=crd)
    data.face = face

    return data


def face_to_tri(face: List[str]) -> Tensor:
    face_index = [[int(x) for x in line.strip().split()] for line in face]

    triangle = Tensor([line[1:] for line in face_index if line[0] == 3]).long()

    rect = Tensor([line[1:] for line in face_index if line[0] == 4]).long()

    if rect.numel() > 0:
        first, second = rect[:, [0, 1, 2]], rect[:, [0, 2, 3]]
        if triangle.numel() > 0:
            return ops.cat([triangle, first, second], axis=0).T
        else:
            return ops.cat([first, second], axis=0).T
    return triangle.T

