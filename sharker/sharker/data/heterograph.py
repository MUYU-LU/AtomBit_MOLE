import copy
import re
import warnings
from collections import defaultdict, namedtuple
from collections.abc import Mapping
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from typing_extensions import Self
import mindspore as ms
from mindspore import ops, mint
import numpy as np

from .graph import Graph, size_repr, warn_or_raise
from .storage import BaseStorage, EdgeStorage, NodeStorage

from ..utils import (
    bipartite_subgraph,
    contains_isolated_nodes,
    is_undirected,
    mask_select,
)

NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]
DEFAULT_REL = "to"
EDGE_TYPE_STR_SPLIT = "__"


class HeteroGraph(Graph):
    r"""A data object describing a heterogeneous graph, holding multiple node
    and/or edge types in disjunct storage objects.
    Storage objects can hold either node-level, link-level or graph-level
    attributes.
    In general, :class:`~sharker.data.HeteroGraph` tries to mimic the
    behavior of a regular **nested** :python:`Python` dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic MindSpore tensor functionalities.

    .. code-block::

        from sharker.data import HeteroGraph

        data = HeteroGraph()

        # Create two node types "paper" and "author" holding a feature matrix:
        data['paper'].x = ops.randn(num_papers, num_paper_features)
        data['author'].x = ops.randn(num_authors, num_authors_features)

        # Create an edge type "(author, writes, paper)" and building the
        # graph connectivity:
        data['author', 'writes', 'paper'].edge_index = ...  # [2, num_edges]

        data['paper'].num_nodes
        >>> 23

        data['author', 'writes', 'paper'].num_edges
        >>> 52

        # MindSpore tensor functionality:
        data = data.pin_memory()

    Note that there exists multiple ways to create a heterogeneous graph data,
    *e.g.*:

    * To initialize a node of type :obj:`"paper"` holding a node feature
      matrix :obj:`x_paper` named :obj:`x`:

      .. code-block:: python

        from sharker.data import HeteroGraph

        # (1) Assign attributes after initialization,
        data = HeteroGraph()
        data['paper'].x = x_paper

        # or (2) pass them as keyword arguments during initialization,
        data = HeteroGraph(paper={ 'x': x_paper })

        # or (3) pass them as dictionaries during initialization,
        data = HeteroGraph({'paper': { 'x': x_paper }})

    * To initialize an edge from source node type :obj:`"author"` to
      destination node type :obj:`"paper"` with relation type :obj:`"writes"`
      holding a graph connectivity matrix :obj:`edge_index_author_paper` named
      :obj:`edge_index`:

      .. code-block:: python

        # (1) Assign attributes after initialization,
        data = HeteroGraph()
        data['author', 'writes', 'paper'].edge_index = edge_index_author_paper

        # or (2) pass them as keyword arguments during initialization,
        data = HeteroGraph(author__writes__paper={
            'edge_index': edge_index_author_paper
        })

        # or (3) pass them as dictionaries during initialization,
        data = HeteroGraph({
            ('author', 'writes', 'paper'):
            { 'edge_index': edge_index_author_paper }
        })
    """

    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()

        self.__dict__["_store"] = BaseStorage(_parent=self)
        self.__dict__["_node_store_dict"] = {}
        self.__dict__["_edge_store_dict"] = {}

        for key, value in chain((_mapping or {}).items(), kwargs.items()):
            if "__" in key and isinstance(value, Mapping):
                key = tuple(key.split("__"))

            if isinstance(value, Mapping):
                self[key].update(value)
            else:
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]) -> Self:
        r"""Creates a :class:`~sharker.data.HeteroGraph` object from a
        dictionary.
        """
        out = cls()
        for key, value in mapping.items():
            if key == "_store":
                out.__dict__["_store"] = BaseStorage(_parent=out, **value)
            elif isinstance(key, str):
                out._node_store_dict[key] = NodeStorage(_parent=out, _key=key, **value)
            else:
                out._edge_store_dict[key] = EdgeStorage(_parent=out, _key=key, **value)
        return out

    def __getattr__(self, key: str) -> Any:
        # `data.*_dict` => Link to node and edge stores.
        # `data.*` => Link to the `_store`.
        # Using `data.*_dict` is the same as using `collect()` for collecting
        # nodes and edges features.
        if hasattr(self._store, key):
            return getattr(self._store, key)
        elif bool(re.search("_dict$", key)):
            return self.collect(key[:-5])
        raise AttributeError(
            f"'{self.__class__.__name__}' has no " f"attribute '{key}'"
        )

    def __setattr__(self, key: str, value: Any):
        # NOTE: We aim to prevent duplicates in node or edge types.
        if key in self.node_types:
            raise AttributeError(f"'{key}' is already present as a node type")
        elif key in self.edge_types:
            raise AttributeError(f"'{key}' is already present as an edge type")
        setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    def __getitem__(
        self, *args: Union[str, Tuple[str, str], Tuple[str, str, str]]
    ) -> Any:
        # `data[*]` => Link to either `_store`, _node_store_dict` or
        # `_edge_store_dict`.
        # If neither is present, we create a new `Storage` object for the given
        # node/edge-type.
        key = self._to_canonical(*args)

        out = self._store.get(key, None)
        if out is not None:
            return out

        if isinstance(key, tuple):
            return self.get_edge_store(*key)
        else:
            return self.get_node_store(key)

    def __setitem__(self, key: str, value: Any):
        if key in self.node_types:
            raise AttributeError(f"'{key}' is already present as a node type")
        elif key in self.edge_types:
            raise AttributeError(f"'{key}' is already present as an edge type")
        self._store[key] = value

    def __delitem__(self, *args: Union[str, Tuple[str, str], Tuple[str, str, str]]):
        key = self._to_canonical(*args)
        if key in self.edge_types:
            del self._edge_store_dict[key]
        elif key in self.node_types:
            del self._node_store_dict[key]
        else:
            del self._store[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__["_store"] = copy.copy(self._store)
        out._store._parent = out
        out.__dict__["_node_store_dict"] = {}
        for key, store in self._node_store_dict.items():
            out._node_store_dict[key] = copy.copy(store)
            out._node_store_dict[key]._parent = out
        out.__dict__["_edge_store_dict"] = {}
        for key, store in self._edge_store_dict.items():
            out._edge_store_dict[key] = copy.copy(store)
            out._edge_store_dict[key]._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        for key in self._node_store_dict.keys():
            out._node_store_dict[key]._parent = out
        for key in out._edge_store_dict.keys():
            out._edge_store_dict[key]._parent = out
        return out

    def __repr__(self) -> str:
        info1 = [size_repr(k, v, 2) for k, v in self._store.items()]
        info2 = [size_repr(k, v, 2) for k, v in self._node_store_dict.items()]
        info3 = [size_repr(k, v, 2) for k, v in self._edge_store_dict.items()]
        info = ",\n".join(info1 + info2 + info3)
        info = f"\n{info}\n" if len(info) > 0 else info
        return f"{self.__class__.__name__}({info})"

    def stores_as(self, data: Self):
        for node_type in data.node_types:
            self.get_node_store(node_type)
        for edge_type in data.edge_types:
            self.get_edge_store(*edge_type)
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        r"""Returns a list of all storages of the graph."""
        return [self._store] + list(self.node_stores) + list(self.edge_stores)

    @property
    def node_types(self) -> List[str]:
        r"""Returns a list of all node types of the graph."""
        return list(self._node_store_dict.keys())

    @property
    def node_stores(self) -> List[NodeStorage]:
        r"""Returns a list of all node storages of the graph."""
        return list(self._node_store_dict.values())

    @property
    def edge_types(self) -> List[Tuple[str, str, str]]:
        r"""Returns a list of all edge types of the graph."""
        return list(self._edge_store_dict.keys())

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        r"""Returns a list of all edge storages of the graph."""
        return list(self._edge_store_dict.values())

    def node_items(self) -> List[Tuple[str, NodeStorage]]:
        r"""Returns a list of node type and node storage pairs."""
        return list(self._node_store_dict.items())

    def edge_items(self) -> List[Tuple[Tuple[str, str, str], EdgeStorage]]:
        r"""Returns a list of edge type and edge storage pairs."""
        return list(self._edge_store_dict.items())

    def to_dict(self) -> Dict[str, Any]:
        out_dict: Dict[str, Any] = {}
        out_dict["_store"] = self._store.to_dict()
        for key, store in chain(
            self._node_store_dict.items(), self._edge_store_dict.items()
        ):
            out_dict[key] = store.to_dict()
        return out_dict

    def to_namedtuple(self) -> NamedTuple:
        field_names = list(self._store.keys())
        field_values = list(self._store.values())
        field_names += [
            "__".join(key) if isinstance(key, tuple) else key
            for key in self.node_types + self.edge_types
        ]
        field_values += [
            store.to_namedtuple() for store in self.node_stores + self.edge_stores
        ]
        DataTuple = namedtuple("DataTuple", field_names)
        return DataTuple(*field_values)

    def set_value_dict(
        self,
        key: str,
        value_dict: Dict[str, Any],
    ) -> Self:
        r"""Sets the values in the dictionary :obj:`value_dict` to the
        attribute with name :obj:`key` to all node/edge types present in the
        dictionary.

        .. code-block:: python

           data = HeteroGraph()

           data.set_value_dict('x', {
               'paper': ops.randn(4, 16),
               'author': ops.randn(8, 32),
           })

           print(data['paper'].x)
        """
        for k, v in (value_dict or {}).items():
            self[k][key] = v
        return self

    def update(self, data: Self) -> Self:
        for store in data.stores:
            for key, value in store.items():
                self[store._key][key] = value
        return self

    def __cat_dim__(
        self,
        key: str,
        value: Any,
        store: Optional[NodeOrEdgeStorage] = None,
        *args,
        **kwargs,
    ) -> Any:
        if isinstance(store, EdgeStorage) and "index" in key:
            return -1
        return 0

    def __inc__(
        self,
        key: str,
        value: Any,
        store: Optional[NodeOrEdgeStorage] = None,
        *args,
        **kwargs,
    ) -> Any:
        if "batch" in key and isinstance(value, (ms.Tensor, np.array)):
            return int(value.max()) + 1
        elif isinstance(store, EdgeStorage) and "index" in key:
            if isinstance(value, ms.Tensor):
                return ms.Tensor(store.shape).view(2, 1)
            elif isinstance(value, np.ndarray):
                return np.array(store.shape).reshape(2, 1)
        else:
            return 0

    @property
    def num_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes in the graph."""
        return super().num_nodes

    @property
    def num_node_features(self) -> Dict[str, int]:
        r"""Returns the number of features per node type in the graph."""
        return {
            key: store.num_node_features for key, store in self._node_store_dict.items()
        }

    @property
    def num_features(self) -> Dict[str, int]:
        r"""Returns the number of features per node type in the graph.
        Alias for :py:attr:`~num_node_features`.
        """
        return self.num_node_features

    @property
    def num_edge_features(self) -> Dict[Tuple[str, str, str], int]:
        r"""Returns the number of features per edge type in the graph."""
        return {
            key: store.num_edge_features for key, store in self._edge_store_dict.items()
        }

    def has_isolated_nodes(self) -> bool:
        r"""Returns :obj:`True` if the graph contains isolated nodes."""
        edge_index, _, _ = to_homogeneous_edge_index(self)
        return contains_isolated_nodes(edge_index, num_nodes=self.num_nodes)

    def is_undirected(self) -> bool:
        r"""Returns :obj:`True` if graph edges are undirected."""
        edge_index, _, _ = to_homogeneous_edge_index(self)
        return is_undirected(edge_index, num_nodes=self.num_nodes)

    def validate(self, raise_on_error: bool = True) -> bool:
        r"""Validates the correctness of the data."""
        cls_name = self.__class__.__name__
        status = True

        node_types = set(self.node_types)
        num_src_node_types = {src for src, _, _ in self.edge_types}
        num_dst_node_types = {dst for _, _, dst in self.edge_types}

        dangling_types = (num_src_node_types | num_dst_node_types) - node_types
        if len(dangling_types) > 0:
            status = False
            warn_or_raise(
                f"The node types {dangling_types} are referenced in edge "
                f"types but do not exist as node types",
                raise_on_error,
            )

        dangling_types = node_types - (num_src_node_types | num_dst_node_types)
        if len(dangling_types) > 0:
            warn_or_raise(
                f"The node types {dangling_types} are isolated and are not "
                f"referenced by any edge type ",
                raise_on_error=False,
            )

        for edge_type, store in self._edge_store_dict.items():
            src, _, dst = edge_type

            num_src_nodes = self[src].num_nodes
            num_dst_nodes = self[dst].num_nodes
            if num_src_nodes is None:
                status = False
                warn_or_raise(
                    f"'num_nodes' is undefined in node type '{src}' of "
                    f"'{cls_name}'",
                    raise_on_error,
                )

            if num_dst_nodes is None:
                status = False
                warn_or_raise(
                    f"'num_nodes' is undefined in node type '{dst}' of "
                    f"'{cls_name}'",
                    raise_on_error,
                )

            if "edge_index" in store:
                if store.edge_index.dim() != 2 or store.edge_index.shape[0] != 2:
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} needs to be "
                        f"of shape [2, num_edges] in '{cls_name}' (found "
                        f"{store.edge_index.shape})",
                        raise_on_error,
                    )

            if "edge_index" in store and store.edge_index.numel() > 0:
                if store.edge_index.min() < 0:
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} contains "
                        f"negative indices in '{cls_name}' "
                        f"(found {int(store.edge_index.min())})",
                        raise_on_error,
                    )

                if (
                    num_src_nodes is not None
                    and store.edge_index[0].max() >= num_src_nodes
                ):
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} contains "
                        f"larger source indices than the number of nodes "
                        f"({num_src_nodes}) of this node type in '{cls_name}' "
                        f"(found {int(store.edge_index[0].max())})",
                        raise_on_error,
                    )

                if (
                    num_dst_nodes is not None
                    and store.edge_index[1].max() >= num_dst_nodes
                ):
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} contains "
                        f"larger destination indices than the number of nodes "
                        f"({num_dst_nodes}) of this node type in '{cls_name}' "
                        f"(found {int(store.edge_index[1].max())})",
                        raise_on_error,
                    )

        return status

    def debug(self):
        pass  # TODO

    ###########################################################################

    def _to_canonical(
        self, *args: Union[str, Tuple[str, str], Tuple[str, str, str]]
    ) -> Union[str, Tuple[str, str, str]]:
        # Converts a given `QueryType` to its "canonical type":
        # 1. `relation_type` will get mapped to the unique
        #    `(src_node_type, relation_type, dst_node_type)` tuple.
        # 2. `(src_node_type, dst_node_type)` will get mapped to the unique
        #    `(src_node_type, *, dst_node_type)` tuple, and
        #    `(src_node_type, 'to', dst_node_type)` otherwise.
        if len(args) == 1:
            args = args[0]

        if isinstance(args, str):
            node_types = [key for key in self.node_types if key == args]
            if len(node_types) == 1:
                args = node_types[0]
                return args

            # Try to map to edge type based on unique relation type:
            edge_types = [key for key in self.edge_types if key[1] == args]
            if len(edge_types) == 1:
                args = edge_types[0]
                return args

        elif len(args) == 2:
            # Try to find the unique source/destination node tuple:
            edge_types = [
                key
                for key in self.edge_types
                if key[0] == args[0] and key[-1] == args[-1]
            ]
            if len(edge_types) == 1:
                args = edge_types[0]
                return args
            elif len(edge_types) == 0:
                args = (args[0], DEFAULT_REL, args[1])
                return args

        return args

    def metadata(self) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        r"""Returns the heterogeneous meta-data, *i.e.* its node and edge
        types.

        .. code-block:: python

            data = HeteroGraph()
            data['paper'].x = ...
            data['author'].x = ...
            data['author', 'writes', 'paper'].edge_index = ...

            print(data.metadata())
            >>> (['paper', 'author'], [('author', 'writes', 'paper')])
        """
        return self.node_types, self.edge_types

    def collect(
        self,
        key: str,
        allow_empty: bool = False,
    ) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        r"""Collects the attribute :attr:`key` from all node and edge types.

        .. code-block:: python

            data = HeteroGraph()
            data['paper'].x = ...
            data['author'].x = ...

            print(data.collect('x'))
            >>> { 'paper': ..., 'author': ...}

        .. note::

            This is equivalent to writing :obj:`data.x_dict`.

        Args:
            key (str): The attribute to collect from all node and ege types.
            allow_empty (bool, optional): If set to :obj:`True`, will not raise
                an error in case the attribute does not exit in any node or
                edge type. (default: :obj:`False`)
        """
        mapping = {}
        for subtype, store in chain(
            self._node_store_dict.items(), self._edge_store_dict.items()
        ):
            if hasattr(store, key):
                mapping[subtype] = getattr(store, key)
        if not allow_empty and len(mapping) == 0:
            raise KeyError(
                f"Tried to collect '{key}' but did not find any "
                f"occurrences of it in any node and/or edge type"
            )
        return mapping

    def _check_type_name(self, name: str):
        if "__" in name:
            warnings.warn(
                f"The type '{name}' contains double underscores "
                f"('__') which may lead to unexpected behavior. "
                f"To avoid any issues, ensure that your type names "
                f"only contain single underscores."
            )

    def get_node_store(self, key: str) -> NodeStorage:
        r"""Gets the :class:`~sharker.data.storage.NodeStorage` object
        of a particular node type :attr:`key`.
        If the storage is not present yet, will create a new
        :class:`sharker.data.storage.NodeStorage` object for the given
        node type.

        .. code-block:: python

            data = HeteroGraph()
            node_storage = data.get_node_store('paper')
        """
        out = self._node_store_dict.get(key, None)
        if out is None:
            self._check_type_name(key)
            out = NodeStorage(_parent=self, _key=key)
            self._node_store_dict[key] = out
        return out

    def get_edge_store(self, src: str, rel: str, dst: str) -> EdgeStorage:
        r"""Gets the :class:`~sharker.data.storage.EdgeStorage` object
        of a particular edge type given by the tuple :obj:`(src, rel, dst)`.
        If the storage is not present yet, will create a new
        :class:`sharker.data.storage.EdgeStorage` object for the given
        edge type.

        .. code-block:: python

            data = HeteroGraph()
            edge_storage = data.get_edge_store('author', 'writes', 'paper')
        """
        key = (src, rel, dst)
        out = self._edge_store_dict.get(key, None)
        if out is None:
            self._check_type_name(rel)
            out = EdgeStorage(_parent=self, _key=key)
            self._edge_store_dict[key] = out
        return out

    def rename(self, name: str, new_name: str) -> Self:
        r"""Renames the node type :obj:`name` to :obj:`new_name` in-place."""
        node_store = self._node_store_dict.pop(name)
        node_store._key = new_name
        self._node_store_dict[new_name] = node_store

        for edge_type in self.edge_types:
            src, rel, dst = edge_type
            if src == name or dst == name:
                edge_store = self._edge_store_dict.pop(edge_type)
                src = new_name if src == name else src
                dst = new_name if dst == name else dst
                edge_type = (src, rel, dst)
                edge_store._key = edge_type
                self._edge_store_dict[edge_type] = edge_store

        return self

    def subgraph(self, subset_dict: Dict[str, ms.Tensor]) -> Self:
        r"""Returns the induced subgraph containing the node types and
        corresponding nodes in :obj:`subset_dict`.

        If a node type is not a key in :obj:`subset_dict` then all nodes of
        that type remain in the graph.

        .. code-block:: python

            data = HeteroGraph()
            data['paper'].x = ...
            data['author'].x = ...
            data['conference'].x = ...
            data['paper', 'cites', 'paper'].edge_index = ...
            data['author', 'paper'].edge_index = ...
            data['paper', 'conference'].edge_index = ...
            print(data)
            >>> HeteroGraph(
                paper={ x=[10, 16] },
                author={ x=[5, 32] },
                conference={ x=[5, 8] },
                (paper, cites, paper)={ edge_index=[2, 50] },
                (author, to, paper)={ edge_index=[2, 30] },
                (paper, to, conference)={ edge_index=[2, 25] }
            )

            subset_dict = {
                'paper': Tensor, 5, 6]),
                'author': Tensor([0, 2]),
            }

            print(data.subgraph(subset_dict))
            >>> HeteroGraph(
                paper={ x=[4, 16] },
                author={ x=[2, 32] },
                conference={ x=[5, 8] },
                (paper, cites, paper)={ edge_index=[2, 24] },
                (author, to, paper)={ edge_index=[2, 5] },
                (paper, to, conference)={ edge_index=[2, 10] }
            )

        Args:
            subset_dict (Dict[str, LongTensor or BoolTensor]): A dictionary
                holding the nodes to keep for each node type.
        """
        data = copy.copy(self)
        subset_dict = copy.copy(subset_dict)

        for node_type, subset in subset_dict.items():
            for key, value in self[node_type].items():
                if key == "num_nodes":
                    if subset.dtype == ms.bool_:
                        data[node_type].num_nodes = int(subset.sum())
                    else:
                        data[node_type].num_nodes = subset.shape[0]
                elif self[node_type].is_node_attr(key):
                    data[node_type][key] = value[subset]
                else:
                    data[node_type][key] = value

        for edge_type in self.edge_types:
            if "edge_index" not in self[edge_type]:
                continue

            src, _, dst = edge_type

            src_subset = subset_dict.get(src)
            if src_subset is None:
                src_subset = mint.arange(data[src].num_nodes)
            dst_subset = subset_dict.get(dst)
            if dst_subset is None:
                dst_subset = mint.arange(data[dst].num_nodes)

            edge_index, _, edge_mask = bipartite_subgraph(
                (src_subset, dst_subset),
                self[edge_type].edge_index,
                relabel_nodes=True,
                size=(self[src].num_nodes, self[dst].num_nodes),
                return_edge_mask=True,
            )

            for key, value in self[edge_type].items():
                if key == "edge_index":
                    data[edge_type].edge_index = edge_index
                elif self[edge_type].is_edge_attr(key):
                    data[edge_type][key] = value[edge_mask]
                else:
                    data[edge_type][key] = value

        return data

    def edge_subgraph(
        self,
        subset_dict: Dict[Tuple[str, str, str], ms.Tensor],
    ) -> Self:
        r"""Returns the induced subgraph given by the edge indices in
        :obj:`subset_dict` for certain edge types.
        Will currently preserve all the nodes in the graph, even if they are
        isolated after subgraph computation.

        Args:
            subset_dict (Dict[Tuple[str, str, str], LongTensor or BoolTensor]):
                A dictionary holding the edges to keep for each edge type.
        """
        data = copy.copy(self)

        for edge_type, subset in subset_dict.items():
            edge_store, new_edge_store = self[edge_type], data[edge_type]
            for key, value in edge_store.items():
                if edge_store.is_edge_attr(key):
                    dim = self.__cat_dim__(key, value, edge_store)
                    if subset.dtype == ms.bool_:
                        new_edge_store[key] = mask_select(value, dim, subset)
                    else:
                        new_edge_store[key] = value.index_select(dim, subset)

        return data

    def node_type_subgraph(self, node_types: List[str]) -> Self:
        r"""Returns the subgraph induced by the given :obj:`node_types`, *i.e.*
        the returned :class:`HeteroGraph` object only contains the node types
        which are included in :obj:`node_types`, and only contains the edge
        types where both end points are included in :obj:`node_types`.
        """
        data = copy.copy(self)
        for edge_type in self.edge_types:
            src, _, dst = edge_type
            if src not in node_types or dst not in node_types:
                del data[edge_type]
        for node_type in self.node_types:
            if node_type not in node_types:
                del data[node_type]
        return data

    def edge_type_subgraph(self, edge_types: List[str]) -> Self:
        r"""Returns the subgraph induced by the given :obj:`edge_types`, *i.e.*
        the returned :class:`HeteroGraph` object only contains the edge types
        which are included in :obj:`edge_types`, and only contains the node
        types of the end points which are included in :obj:`node_types`.
        """
        edge_types = [self._to_canonical(e) for e in edge_types]

        data = copy.copy(self)
        for edge_type in self.edge_types:
            if edge_type not in edge_types:
                del data[edge_type]
        node_types = set(e[0] for e in edge_types)
        node_types |= set(e[-1] for e in edge_types)
        for node_type in self.node_types:
            if node_type not in node_types:
                del data[node_type]
        return data

    def to_homogeneous(
        self,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        add_node_type: bool = True,
        add_edge_type: bool = True,
        dummy_values: bool = True,
    ) -> Graph:
        """Converts a :class:`~sharker.data.HeteroGraph` object to a
        homogeneous :class:`~sharker.data.Graph` object.
        By default, all features with same feature dimensionality across
        different types will be merged into a single representation, unless
        otherwise specified via the :obj:`node_attrs` and :obj:`edge_attrs`
        arguments.
        Furthermore, attributes named :obj:`node_type` and :obj:`edge_type`
        will be added to the returned :class:`~sharker.data.Graph`
        object, denoting node-level and edge-level vectors holding the
        node and edge type as integers, respectively.

        Args:
            node_attrs (List[str], optional): The node features to combine
                across all node types. These node features need to be of the
                same feature dimensionality. If set to :obj:`None`, will
                automatically determine which node features to combine.
                (default: :obj:`None`)
            edge_attrs (List[str], optional): The edge features to combine
                across all edge types. These edge features need to be of the
                same feature dimensionality. If set to :obj:`None`, will
                automatically determine which edge features to combine.
                (default: :obj:`None`)
            add_node_type (bool, optional): If set to :obj:`False`, will not
                add the node-level vector :obj:`node_type` to the returned
                :class:`~sharker.data.Graph` object.
                (default: :obj:`True`)
            add_edge_type (bool, optional): If set to :obj:`False`, will not
                add the edge-level vector :obj:`edge_type` to the returned
                :class:`~sharker.data.Graph` object.
                (default: :obj:`True`)
            dummy_values (bool, optional): If set to :obj:`True`, will fill
                attributes of remaining types with dummy values.
                Dummy values are :obj:`NaN` for floating point attributes,
                :obj:`False` for booleans, and :obj:`-1` for integers.
                (default: :obj:`True`)
        """

        def get_sizes(stores: List[BaseStorage]) -> Dict[str, List[Tuple]]:
            sizes_dict = defaultdict(list)
            for store in stores:
                for key, value in store.items():
                    if key in ["edge_index", "edge_label_index", "adj", "adj_t"]:
                        continue
                    if isinstance(value, ms.Tensor):
                        dim = self.__cat_dim__(key, value, store)
                        size = value.shape[:dim] + value.shape[dim+1:]
                        sizes_dict[key].append(tuple(size))
            return sizes_dict

        def fill_dummy_(stores: List[BaseStorage], keys: Optional[List[str]] = None):
            sizes_dict = get_sizes(stores)

            if keys is not None:
                sizes_dict = {
                    key: sizes for key, sizes in sizes_dict.items() if key in keys
                }

            sizes_dict = {
                key: sizes for key, sizes in sizes_dict.items() if len(set(sizes)) == 1
            }

            for store in stores:  # Fill stores with dummy features:
                for key, sizes in sizes_dict.items():
                    if key not in store:
                        ref = list(self.collect(key).values())[0]
                        dim = self.__cat_dim__(key, ref, store)
                        if ref.is_floating_point():
                            dummy = float("NaN")
                        elif ref.dtype == ms.bool_:
                            dummy = False
                        else:
                            dummy = -1
                        if isinstance(store, NodeStorage):
                            dim_size = store.num_nodes
                        else:
                            dim_size = store.num_edges
                        shape = sizes[0][:dim] + (dim_size,) + sizes[0][dim:]
                        store[key] = ops.full(shape, dummy, dtype=ref.dtype)

        def _consistent_size(stores: List[BaseStorage]) -> List[str]:
            sizes_dict = get_sizes(stores)
            keys = []
            for key, sizes in sizes_dict.items():
                if len(sizes) != len(stores):
                    continue
                lengths = set([len(size) for size in sizes])
                if len(lengths) != 1:
                    continue
                if len(sizes[0]) != 1 and len(set(sizes)) != 1:
                    continue
                keys.append(key)
            return keys

        if dummy_values:
            self = copy.copy(self)
            fill_dummy_(self.node_stores, node_attrs)
            fill_dummy_(self.edge_stores, edge_attrs)

        edge_index, node_slices, edge_slices = to_homogeneous_edge_index(self)

        data = Graph(**self._store.to_dict())
        if edge_index is not None:
            data.edge_index = edge_index
        data._node_type_names = list(node_slices.keys())
        data._edge_type_names = list(edge_slices.keys())

        # Combine node attributes into a single tensor:
        if node_attrs is None:
            node_attrs = _consistent_size(self.node_stores)
        for key in node_attrs:
            if key in {"ptr"}:
                continue
            values = [store[key] for store in self.node_stores]
            dim = self.__cat_dim__(key, values[0], self.node_stores[0])
            dim = values[0].dim() + dim if dim < 0 else dim
            # For two-dimensional features, we allow arbitrary shapes and
            # pad them with zeros if necessary in case their size doesn't
            # match:
            if values[0].dim() == 2 and dim == 0:
                _max = max([value.shape[-1] for value in values])
                for i, v in enumerate(values):
                    if v.shape[-1] < _max:
                        pad = v.new_zeros([v.shape[0], _max - v.shape[-1]])
                        values[i] = mint.cat([v, pad], dim=-1)
            value = mint.cat(values, dim=dim)
            data[key] = value

        if not data.can_infer_num_nodes:
            data.num_nodes = list(node_slices.values())[-1][1]

        # Combine edge attributes into a single tensor:
        if edge_attrs is None:
            edge_attrs = _consistent_size(self.edge_stores)
        for key in edge_attrs:
            values = [store[key] for store in self.edge_stores]
            dim = self.__cat_dim__(key, values[0], self.edge_stores[0])
            value = mint.cat(values, dim=dim) if len(values) > 1 else values[0]
            data[key] = value

        if "edge_label_index" in self:
            edge_label_index_dict = self.edge_label_index_dict
            for edge_type, edge_label_index in edge_label_index_dict.items():
                edge_label_index = edge_label_index.copy()
                edge_label_index[0] += node_slices[edge_type[0]][0]
                edge_label_index[1] += node_slices[edge_type[-1]][0]
                edge_label_index_dict[edge_type] = edge_label_index
            data.edge_label_index = mint.cat(
                list(edge_label_index_dict.values()), dim=-1
            )

        if add_node_type:
            sizes = [offset[1] - offset[0] for offset in node_slices.values()]
            sizes = ms.Tensor(sizes, dtype=ms.int64)
            node_type = mint.arange(len(sizes))
            data.node_type = node_type.repeat_interleave(sizes)

        if add_edge_type and edge_index is not None:
            sizes = [offset[1] - offset[0] for offset in edge_slices.values()]
            sizes = ms.Tensor(sizes, dtype=ms.int64)
            edge_type = mint.arange(len(sizes))
            data.edge_type = edge_type.repeat_interleave(sizes)

        return data


# Helper functions ############################################################


def get_node_slices(num_nodes: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    r"""Returns the boundaries of each node type in a graph."""
    node_slices: Dict[str, Tuple[int, int]] = {}
    cumsum = 0
    for node_type, N in num_nodes.items():
        node_slices[node_type] = (cumsum, cumsum + N)
        cumsum += N
    return node_slices


def offset_edge_index(
    node_slices: Dict[str, Tuple[int, int]],
    edge_type: Tuple[str, str, str],
    edge_index: ms.Tensor,
) -> ms.Tensor:
    r"""Increases the edge indices by the offsets of source and destination
    node types.
    """
    src, _, dst = edge_type
    offset = [[node_slices[src][0]], [node_slices[dst][0]]]
    offset = ms.Tensor(offset)
    return edge_index + offset


def to_homogeneous_edge_index(
    graph: HeteroGraph,
) -> Tuple[Optional[ms.Tensor], Dict[str, Any], Dict[Tuple[str, str, str], Any]]:
    r"""Converts a heterogeneous graph into a homogeneous typed graph."""
    # Record slice information per node type:
    node_slices = get_node_slices(graph.num_nodes_dict)

    # Record edge indices and slice information per edge type:
    cumsum = 0
    edge_indices: List[ms.Tensor] = []
    edge_slices: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    for edge_type, edge_index in graph.collect("edge_index", True).items():
        edge_index = offset_edge_index(node_slices, edge_type, edge_index)
        edge_indices.append(edge_index)
        edge_slices[edge_type] = (cumsum, cumsum + edge_index.shape[1])
        cumsum += edge_index.shape[1]

    edge_index: Optional[ms.Tensor] = None
    if len(edge_indices) == 1:
        edge_index = edge_indices[0]
    elif len(edge_indices) > 1:
        edge_index = mint.cat(edge_indices, dim=-1)

    return edge_index, node_slices, edge_slices
