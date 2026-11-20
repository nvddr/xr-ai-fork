# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Dialogue``: an output (a ``xr_ai_app.Sink``) that keeps the dialogue log, what was said on both sides.

Speech itself (speech to text, the voice gate, text to speech, barge-in) is the app's voice pipeline, configured by
``app.yaml``'s ``voice:`` section and the profile's ``speech_in`` and ``speech_out`` roles. A ``Dialogue`` speaks
through the participant's default output and keeps ``dialogue.log``, a participant-scoped context log named
``dialogue`` of what the user said (each turn, recorded in :meth:`Dialogue.before_turn`, so the app's turn handler
already finds it there) and what was spoken through this output. It is the only writer of that log: a direct
``p.say`` or ``p.reply`` is not recorded.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import TYPE_CHECKING

from xr_ai_app import Component, ContextLog
from xr_ai_app.trace import note_output

if TYPE_CHECKING:
    from xr_ai_app import Participant, Role, Turn

__all__ = ["Dialogue"]


class Dialogue(Component):
    """Speech out, and the dialogue log (``dialogue.log``).

    ``summarize`` compacts the log when it outgrows ``budget_bytes`` (without it the oldest entries are dropped).
    ``name`` names this output in the trace. Pass ``dialogue.log`` where a component reads the dialogue
    (``context=[dialogue.log]``, ``history=dialogue.log``, ``SopRunner(conversation=dialogue.log)``).
    """

    def __init__(
        self,
        *,
        summarize: Role | None = None,
        budget_bytes: int = 8_000,
        name: str = "voice",
    ) -> None:
        super().__init__()
        self._name = name
        self.log = ContextLog("dialogue", scope="participant", budget_bytes=budget_bytes, summarize=summarize)

    @property
    def name(self) -> str:
        return self._name

    async def before_turn(self, turn: Turn, p: Participant) -> None:
        """Record the user's words (``user: ...``) before the app's turn handler runs."""
        await self.log.publish(p, f"user: {turn.text}")

    async def say(self, p: Participant, text: str, *, urgent: bool = False, source: str | None = None) -> None:
        """Speak ``text`` on the participant's default output, then record it (``assistant: ...``)."""
        note_output(self._name, "say", text)
        await p.say(text, urgent=urgent, source=source)
        await self.log.publish(p, f"assistant: {text}")

    async def reply(self, p: Participant, text: str | AsyncIterable[str]) -> str:
        """Answer the current turn (streaming allowed); record what was spoken when it is not blank."""
        if isinstance(text, str):
            if text:
                note_output(self._name, "reply", text)
            spoken = await p.reply(text)
        else:
            spoken = await p.reply(_mark_first(text, lambda: note_output(self._name, "reply", "")))
            if spoken:
                note_output(self._name, "reply_end", spoken)
        if spoken.strip():
            await self.log.publish(p, f"assistant: {spoken}")
        return spoken

    async def withdraw(self, p: Participant, source: str) -> bool:
        """Drop a queued ``say`` tagged ``source``; its entry stays in the log."""
        return await p.withdraw(source)


async def _mark_first(stream: AsyncIterable[str], on_first: Callable[[], None]) -> AsyncIterator[str]:
    """Pass ``stream`` through, calling ``on_first()`` before the first non-empty fragment."""
    first = True
    async for part in stream:
        if first and part:
            first = False
            on_first()
        yield part
