from __future__ import annotations

import sys
from typing import Any

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


_ACTIVE_PROGRESS_STACK: list["_BaseProgress"] = []


def progress_log(message: str) -> None:
    if _ACTIVE_PROGRESS_STACK:
        _ACTIVE_PROGRESS_STACK[-1].write(message)
        return
    print(message, flush=True)


class _BaseProgress:
    def __enter__(self):
        _ACTIVE_PROGRESS_STACK.append(self)
        self.refresh()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def refresh(self) -> None:
        raise NotImplementedError

    def update(self, n: int = 1) -> None:
        raise NotImplementedError

    def set_postfix_str(self, text: str) -> None:
        raise NotImplementedError

    def write(self, message: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        if self in _ACTIVE_PROGRESS_STACK:
            _ACTIVE_PROGRESS_STACK.remove(self)
        if _ACTIVE_PROGRESS_STACK:
            _ACTIVE_PROGRESS_STACK[-1].refresh()


class _TqdmProgress(_BaseProgress):
    def __init__(self, total: int, desc: str, unit: str, leave: bool):
        self._bar = _tqdm(total=total, desc=desc, unit=unit, leave=leave, dynamic_ncols=True)

    def refresh(self) -> None:
        self._bar.refresh()

    def update(self, n: int = 1) -> None:
        self._bar.update(n)

    def set_postfix_str(self, text: str) -> None:
        self._bar.set_postfix_str(text)

    def write(self, message: str) -> None:
        self._bar.write(message)

    def close(self) -> None:
        self._bar.close()
        super().close()


class _BasicProgress(_BaseProgress):
    def __init__(self, total: int, desc: str, unit: str, leave: bool):
        self.total = max(int(total), 0)
        self.desc = desc
        self.unit = unit
        self.leave = leave
        self.current = 0
        self.postfix = ""
        self._closed = False

    def _render_line(self) -> str:
        width = 24
        ratio = 1.0 if self.total <= 0 else min(max(self.current / self.total, 0.0), 1.0)
        filled = int(round(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        suffix = f" {self.postfix}" if self.postfix else ""
        return (
            f"{self.desc}: [{bar}] {self.current}/{self.total} "
            f"{self.unit} ({ratio * 100:5.1f}%){suffix}"
        )

    def refresh(self) -> None:
        if self._closed:
            return
        sys.stderr.write("\r" + self._render_line())
        sys.stderr.flush()

    def update(self, n: int = 1) -> None:
        if self._closed:
            return
        self.current = min(self.total, self.current + max(int(n), 0))
        self.refresh()

    def set_postfix_str(self, text: str) -> None:
        if self._closed:
            return
        self.postfix = text.strip()
        self.refresh()

    def write(self, message: str) -> None:
        if self._closed:
            print(message, flush=True)
            return
        sys.stderr.write("\r" + " " * max(len(self._render_line()), 1) + "\r")
        print(message, flush=True)
        self.refresh()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.leave:
            sys.stderr.write("\r" + self._render_line() + "\n")
        else:
            sys.stderr.write("\r" + " " * max(len(self._render_line()), 1) + "\r")
        sys.stderr.flush()
        super().close()


def progress_bar(
    *,
    total: int,
    desc: str,
    unit: str = "step",
    leave: bool = True,
) -> _BaseProgress:
    if _tqdm is not None:
        return _TqdmProgress(total=total, desc=desc, unit=unit, leave=leave)
    return _BasicProgress(total=total, desc=desc, unit=unit, leave=leave)
