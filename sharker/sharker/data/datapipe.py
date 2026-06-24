import copy
from typing import Any, Callable, Iterator, Sequence
from .batch import Batch

IterDataPipe = IterBatcher = object


def functional_datapipe(name: str) -> Callable:
    return lambda cls: cls


class Batcher:
    def __init__(
        self,
        dp: IterDataPipe,
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dp,
            batch_size=batch_size,
            drop_last=drop_last,
            wrapper_class=Batch.from_data_list,
        )


class DatasetAdapter(IterDataPipe):
    def __init__(self, dataset: Sequence[Any]) -> None:
        super().__init__()
        self.dataset = dataset
        self.range = range(len(self))

    def is_shardable(self) -> bool:
        return True

    def apply_sharding(self, num_shards: int, shard_idx: int) -> None:
        self.range = range(shard_idx, len(self), num_shards)

    def __iter__(self) -> Iterator:
        for i in self.range:
            yield self.dataset[i]

    def __len__(self) -> int:
        return len(self.dataset)


def functional_transform(name: str) -> Callable:
    def wrapper(cls: Any) -> Any:
        @functional_datapipe(name)
        class DynamicMapper(IterDataPipe):
            def __init__(
                self,
                dp: IterDataPipe,
                *args: Any,
                **kwargs: Any,
            ) -> None:
                super().__init__()
                self.dp = dp
                self.fn = cls(*args, **kwargs)

            def __iter__(self) -> Iterator:
                for data in self.dp:
                    yield self.fn(copy.copy(data))

        return cls

    return wrapper
