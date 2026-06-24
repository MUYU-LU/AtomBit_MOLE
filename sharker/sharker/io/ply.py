from mindspore import Tensor, ops, nn

from ..data import Graph

try:
    import openmesh
except ImportError:
    openmesh = None


def read_ply(path: str) -> Graph:
    if openmesh is None:
        raise ImportError("`read_ply` requires the `openmesh` package.")

    mesh = openmesh.read_trimesh(path)
    crd = Tensor.from_numpy(mesh.points()).float()
    face = Tensor.from_numpy(mesh.face_vertex_indices())
    face = face.t().long()
    return Graph(crd=crd, face=face)
