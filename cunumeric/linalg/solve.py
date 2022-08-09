from typing import TYPE_CHECKING

from cunumeric.config import CuNumericOpCode

from legate.core import Rect, types as ty
from legate.core.shape import Shape

from .exception import LinAlgError

if TYPE_CHECKING:
    from legate.core.context import Context
    from legate.core.store import Store, StorePartition

    from ..deferred import DeferredArray
    from ..runtime import Runtime


# TODO (rohany): Not sure what import cycle i'm looking at there it can't find deferred array
def solve(a, b, x) -> None:
    ctx = a.context
    task = ctx.create_task(CuNumericOpCode.DGESV)
    task.add_input(a.base)
    task.add_input(b.base)
    task.add_output(x.base)
    # Restrict this to be a single task for now.
    task.add_broadcast(x.base)
    task.execute()
