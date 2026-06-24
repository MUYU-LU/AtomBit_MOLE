from typing import Iterator, List, Optional, Tuple, Union

from mindspore import Tensor, ops, nn
from ..data import Graph


def yield_file(in_file: str) -> Iterator[Tuple[str, List[Union[int, float]]]]:

    f = open(in_file)
    buf = f.read()
    f.close()
    for b in buf.split("\n"):
        if b.startswith("v "):
            yield "v", [float(x) for x in b.split(" ")[1:]]
        elif b.startswith("f "):
            triangles = b.split(" ")[1:]
            # -1 as .obj is base 1 but the Graph class expects base 0 indices
            yield "f", [int(t.split("/")[0]) - 1 for t in triangles]
        else:
            yield "", []


def read_obj(in_file: str) -> Optional[Graph]:
    vertices = []
    faces = []

    for k, v in yield_file(in_file):
        if k == "v":
            vertices.append(v)
        elif k == "f":
            faces.append(v)

    if not len(faces) or not len(vertices):
        return None

    crd = Tensor(vertices).float()
    face = Tensor(faces).long().t()

    data = Graph(crd=crd, face=face)

    return data
