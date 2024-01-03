
from typing import Any


class Linear:
    def __init__(self) -> None:
        self.w
        self.b
        pass

    def __call__(self, input) -> Any:
        return input @ self.w + self.b