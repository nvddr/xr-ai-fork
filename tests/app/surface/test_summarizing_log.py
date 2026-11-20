# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kit's ``SummarizingLog``: a ``ContextLog`` that folds its oldest entries into a summary with a model role.

What the summary role is sent is pinned word for word, and so is what a model reads back (``view``), through
overflow, a summary still running, a failed summary and ``clear``. The log is still a context channel of the core:
``changes()`` wakes on every ``publish``, which is how a step's escalator follows the dialogue.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from xr_ai_app import ContextLog, Participant
from xr_ai_app._base import _tracing
from xr_ai_app.context import changes
from xr_ai_app.testing import FakeModels
from xr_ai_app.trace import MemorySink, Tracer
from xr_ai_kit import SummarizingLog

SYSTEM_64 = (
    "Fold these notes into the running summary. Keep names, numbers, decisions and anything the user asked "
    "for; drop chit-chat. Answer with the new summary only, at most 64 characters."
)


class Held:
    """A summarize role whose every answer waits until the test releases it (or raises ``error``)."""

    def __init__(self, *answers: str, error: Exception | None = None) -> None:
        self.answers = list(answers)
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.release = asyncio.Event()

    async def ask(self, frame: Any, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"frame": frame, "prompt": prompt, **kwargs})
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.answers.pop(0)


async def _publish(log: ContextLog, p: Participant | None, *texts: str) -> None:
    for text in texts:
        await log.publish(p, text)
    await asyncio.sleep(0)  # a summary that started is running


async def test_a_summarizing_log_is_a_context_log_and_what_the_role_is_sent_is_unchanged() -> None:
    models = FakeModels({"sum": ["the user likes tea", "x" * 100, "short"]})
    log = SummarizingLog("notes", scope="participant", budget_bytes=60, summarize=models.role("sum"))
    assert isinstance(log, ContextLog)
    p = Participant("p")
    await _publish(log, p, *(f"note number {i}" for i in range(6)))
    await log.settle(p)
    first = models.calls("sum")[0]
    assert first.prompt == "New notes:\nnote number 0\nnote number 1\nnote number 2"
    assert (first.system, first.thinking, first.max_tokens, first.frame) == (SYSTEM_64, False, 256, None)
    assert log.summary(p) == "the user likes tea"
    assert log.entries(p) == ["note number 3", "note number 4", "note number 5"]
    assert await log.view(p) == "(earlier, summarized) the user likes tea\nnote number 5"
    await _publish(log, p, "note number 6", "note number 7")
    await log.settle(p)  # the second summary, which leaves the log over budget: a third one
    await log.settle(p)
    assert [c.prompt for c in models.calls("sum")[1:]] == [
        "Running summary:\nthe user likes tea\n\nNew notes:\nnote number 3\nnote number 4\nnote number 5\n"
        "note number 6",
        "Running summary:\n" + "x" * 61 + "...\n\nNew notes:\nnote number 7",  # clipped to max(budget // 4, 64)
    ]
    assert log.summary(p) == "short" and log.entries(p) == []
    assert await log.view(p) == "(earlier, summarized) short"


async def test_changes_wakes_on_publish_and_not_when_a_summary_lands() -> None:
    held = Held("summary")
    log = SummarizingLog("dialogue", scope="participant", budget_bytes=60, summarize=held)
    p = Participant("p")
    follow = changes(p, [log])
    await asyncio.wait_for(anext(follow), 1)  # now
    woken = asyncio.ensure_future(anext(follow))
    await asyncio.sleep(0)
    assert not woken.done()
    await log.publish(p, "user: it's in Fahrenheit")
    await asyncio.wait_for(woken, 1)
    await _publish(log, p, *(f"note number {i}" for i in range(5)))
    await asyncio.wait_for(anext(follow), 1)  # the publishes above, folded into one more turn
    assert len(held.calls) == 1
    woken = asyncio.ensure_future(anext(follow))
    held.release.set()
    await log.settle(p)
    await asyncio.sleep(0)
    assert not woken.done()  # a summary landing is not a change
    woken.cancel()
    with pytest.raises(asyncio.CancelledError):
        await woken
    await follow.aclose()


async def test_each_participant_has_its_own_entries_and_summary() -> None:
    models = FakeModels({"sum": ["p's summary"]})
    log = SummarizingLog("dialogue", scope="participant", budget_bytes=60, summarize=models.role("sum"))
    p, q = Participant("p"), Participant("q")
    await log.publish(q, "q said this")
    await _publish(log, p, *(f"note number {i}" for i in range(6)))
    await log.settle(p)
    await log.settle(q)
    assert log.summary(p) == "p's summary" and log.summary(q) == ""
    assert log.entries(q) == ["q said this"]
    assert await log.view(q) == "q said this"
    with pytest.raises(ValueError, match="pass the participant"):
        await log.publish(None, "no one")


async def test_entries_are_kept_while_a_summary_runs_and_compacted_after_it() -> None:
    held = Held("first", "second")
    log = SummarizingLog("notes", scope="participant", budget_bytes=60, summarize=held)
    p = Participant("p")
    await _publish(log, p, *(f"note number {i}" for i in range(5)))
    assert len(held.calls) == 1  # one summary running, of notes 0 to 2
    await _publish(log, p, "note number 5", "note number 6", "note number 7")
    assert len(held.calls) == 1  # over budget again, but nothing is removed while it runs
    assert log.entries(p) == [f"note number {i}" for i in range(3, 8)]
    assert await log.view(p) == "note number 4\nnote number 5\nnote number 6\nnote number 7"
    held.release.set()
    await log.settle(p)  # the first summary, then the second one it started
    await log.settle(p)
    assert held.calls[0] == {
        "frame": None,
        "prompt": "New notes:\nnote number 0\nnote number 1\nnote number 2",
        "system": SYSTEM_64,
        "thinking": False,
        "max_tokens": 256,
    }
    assert held.calls[1]["prompt"] == (
        "Running summary:\nfirst\n\nNew notes:\nnote number 3\nnote number 4\nnote number 5\nnote number 6"
    )
    assert log.summary(p) == "second"
    assert log.entries(p) == ["note number 7"]
    assert await log.view(p) == "(earlier, summarized) second\nnote number 7"


async def test_a_failed_summary_drops_the_entries_and_traces_the_error() -> None:
    tracer = Tracer()
    memory = tracer.add(MemorySink())
    _tracing.activate(tracer)
    try:
        failing = Held(error=RuntimeError("model down"))
        failing.release.set()
        log = SummarizingLog("notes", scope="participant", budget_bytes=60, summarize=failing)
        p = Participant("p")
        await _publish(log, p, *(f"note number {i}" for i in range(5)))
        await log.settle(p)
        assert log.summary(p) == "" and log.entries(p) == ["note number 3", "note number 4"]
        assert await log.view(p) == "(3 earlier entries dropped)\nnote number 3\nnote number 4"
        errors = [(e.name, e.data) for e in memory.events(kind="error")]
        assert errors == [("notes", {"where": "context summary", "error": "RuntimeError"})]

        # An earlier summary still heads the view after a later failure.
        models = FakeModels({"sum": ["kept", RuntimeError("model down")]})
        log = SummarizingLog("notes", scope="participant", budget_bytes=60, summarize=models.role("sum"))
        p = Participant("q")
        await _publish(log, p, *(f"note number {i}" for i in range(5)))
        await log.settle(p)
        await _publish(log, p, *(f"note number {i}" for i in range(5, 9)))
        await log.settle(p)
        assert len(models.calls("sum")) == 2
        assert log.summary(p) == "kept"
        assert await log.view(p) == "(earlier, summarized) kept\nnote number 7\nnote number 8"
    finally:
        _tracing.deactivate(tracer)


async def test_clear_cancels_a_running_summary_and_starts_over() -> None:
    held = Held("never")
    log = SummarizingLog("notes", scope="participant", budget_bytes=60, summarize=held)
    p = Participant("p")
    await _publish(log, p, *(f"note number {i}" for i in range(5)))
    assert len(held.calls) == 1
    follow = changes(p, [log])
    await anext(follow)
    woken = asyncio.ensure_future(anext(follow))
    await log.clear(p)
    await asyncio.wait_for(woken, 1)  # clear is a change
    await log.settle(p)
    assert log.entries(p) == [] and log.summary(p) == "" and await log.view(p) == ""
    held.release.set()
    await asyncio.sleep(0)
    assert log.summary(p) == ""
    await follow.aclose()
