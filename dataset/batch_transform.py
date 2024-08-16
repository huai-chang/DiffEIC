from typing import Any, overload


class BatchTransform:
    
    @overload
    def __call__(self, batch: Any) -> Any:
        ...


class IdentityBatchTransform(BatchTransform):
    
    def __call__(self, batch: Any) -> Any:
        return batch

