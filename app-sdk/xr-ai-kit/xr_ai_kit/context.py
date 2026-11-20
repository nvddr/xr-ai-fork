# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``SummarizingLog``: a context log that folds its oldest entries into a rolling summary instead of dropping them.

It is a core ``xr_ai_app.ContextLog`` (``render`` shows it, ``changes()`` wakes on each ``publish``) whose one
overridable step, ``compact``, hands the entries that no longer fit to a ``summarize`` role, in the background
(``publish`` never waits). While a summary runs, nothing more is removed; when it lands, the log is compacted again.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from xr_ai_app import ContextLog, trace
from xr_ai_app.context import Scope, clip

if TYPE_CHECKING:
    from xr_ai_app import Participant, Role

__all__ = ["SummarizingLog"]

_SUMMARY_PROMPT = (
    "Fold these notes into the running summary. Keep names, numbers, decisions and anything the user asked "
    "for; drop chit-chat. Answer with the new summary only, at most {limit} characters."
)


@dataclass
class _Summary:
    """One scope's summary state, kept beside the log's entries (in the same state, so it ends with them)."""

    text: str = ""
    dropped: int = 0
    folding: asyncio.Task[None] | None = None


class SummarizingLog(ContextLog):
    """A :class:`~xr_ai_app.ContextLog` whose oldest entries are folded into a summary by ``summarize``.

    When the summary and the entries outgrow ``budget_bytes``, the oldest entries (down to half the budget) go to
    ``summarize.ask`` with the running summary, thinking off and at most 256 tokens; the answer, cut to a quarter of
    the budget (at least 64 bytes), is the new summary, which :meth:`view` shows first as ``(earlier, summarized)``.
    A summary that fails drops those entries (``view`` counts them) and traces the error.
    """

    def __init__(self, name: str, *, scope: Scope, budget_bytes: int, summarize: Role) -> None:
        super().__init__(name, scope=scope, budget_bytes=budget_bytes)
        self._summarize = summarize
        self._app_summary = _Summary()

    def _summary_state(self, p: Participant | None, *, fresh: bool = False) -> _Summary:
        """The summary state of ``p``'s scope (``fresh``: a new one, replacing it)."""
        if self.scope == "app":
            if fresh:
                self._app_summary = _Summary()
            return self._app_summary
        if p is None:
            raise ValueError(f"context channel {self.name!r} has {self.scope} scope: pass the participant")
        state: Any = p.state if self.scope == "participant" else p.session.state
        key = f"_context_summary:{self.name}"
        summary = None if fresh else state.get(key)
        if summary is None:
            summary = state[key] = _Summary()
        return summary

    def summary(self, p: Participant | None) -> str:
        """The running summary of the entries folded so far (``""`` before the first)."""
        return self._summary_state(p).text

    def compact(self, p: Participant | None, entries: list[str]) -> None:
        """Start folding the oldest entries into the summary when the log is over budget (never while one runs)."""
        state = self._summary_state(p)
        if _size(state, entries) <= self.budget_bytes or state.folding is not None:
            return
        target = self.budget_bytes // 2
        old: list[str] = []
        while entries and _size(state, entries) > target:
            old.append(entries.pop(0))
        if not old:
            return
        state.folding = asyncio.get_running_loop().create_task(
            self._fold(p, state, entries, old), name=f"context-fold:{self.name}"
        )

    async def _fold(self, p: Participant | None, state: _Summary, entries: list[str], old: Sequence[str]) -> None:
        limit = max(self.budget_bytes // 4, 64)
        prompt = (f"Running summary:\n{state.text}\n\n" if state.text else "") + "New notes:\n" + "\n".join(old)
        try:
            text = await self._summarize.ask(
                None, prompt, system=_SUMMARY_PROMPT.format(limit=limit), thinking=False, max_tokens=256
            )
            state.text = clip(str(text).strip(), limit)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed summary drops the entries instead
            state.dropped += len(old)
            trace.emit("error", self.name, where="context summary", error=type(exc).__name__)
        finally:
            state.folding = None
        self.compact(p, entries)

    async def settle(self, p: Participant | None) -> None:
        """Wait for a running summary to finish (tests and shutdown)."""
        folding = self._summary_state(p).folding
        if folding is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await folding

    async def clear(self, p: Participant | None) -> None:
        """Cancel a running summary, forget the summary and remove every entry."""
        folding = self._summary_state(p).folding
        if folding is not None:
            folding.cancel()
        self._summary_state(p, fresh=True)
        await super().clear(p)

    async def view(self, p: Participant | None, *, budget_bytes: int | None = None) -> str:
        """The summary (if any; else how many entries were dropped), then the newest entries that fit."""
        state = self._summary_state(p)
        budget = self.budget_bytes if budget_bytes is None else min(budget_bytes, self.budget_bytes)
        head = []
        if state.text:
            head.append(f"(earlier, summarized) {state.text}")
        elif state.dropped:
            head.append(f"({state.dropped} earlier entries dropped)")
        lines: list[str] = []
        used = sum(len(h.encode()) + 1 for h in head)
        for entry in reversed(self.entries(p)):
            size = len(entry.encode()) + 1
            if used + size > budget:
                break
            lines.append(entry)
            used += size
        lines.reverse()
        return clip("\n".join([*head, *lines]), budget)


def _size(state: _Summary, entries: Sequence[str]) -> int:
    return len(state.text.encode()) + sum(len(e.encode()) + 1 for e in entries)
