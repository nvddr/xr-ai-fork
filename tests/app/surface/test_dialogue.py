# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kit's ``Dialogue`` and the core's ``Component.before_turn`` hook it records the user's words in."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from xr_ai_app import Component, ContextLog, Participant, PerConnection, Turn
from xr_ai_app._base.app import App, AppConfig
from xr_ai_app.trace import MemorySink
from xr_ai_kit import Dialogue


def _app() -> App:
    return App(AppConfig(config_dir=Path(".")), enrollment=PerConnection(admit=None))


async def _parts(*parts: str) -> AsyncIterator[str]:
    for part in parts:
        yield part


async def test_the_user_turn_is_in_the_log_before_the_handler_runs() -> None:
    app = _app()
    dialogue = app.add(Dialogue())
    seen: list[list[str]] = []

    @app.on_turn
    async def handle(turn: Turn, p: Participant) -> None:
        seen.append(dialogue.log.entries(p))
        await dialogue.reply(p, f"echo {turn.text}")

    async with app.test_session() as s:
        await s.join()
        await s.say("hello")
        await s.settle()
        await s.say("again")
        await s.settle()
        entries = dialogue.log.entries(s.participant())
    assert isinstance(dialogue.log, ContextLog) and dialogue.log.name == "dialogue"
    assert seen == [["user: hello"], ["user: hello", "assistant: echo hello", "user: again"]]
    assert entries == ["user: hello", "assistant: echo hello", "user: again", "assistant: echo again"]


async def test_before_turn_runs_in_registration_order_and_an_error_ends_the_turn() -> None:
    app = _app()
    order: list[str] = []

    class Seen(Component):
        def __init__(self, name: str, *, fail: bool = False) -> None:
            super().__init__()
            self.name, self.fail = name, fail

        async def before_turn(self, turn: Turn, p: Participant) -> None:
            order.append(f"{self.name}:{turn.text}")
            if self.fail and turn.text == "boom":
                raise RuntimeError("before_turn failed")

    app.add(Seen("a"))
    app.add(Seen("b", fail=True))

    @app.on_turn
    async def handle(turn: Turn, p: Participant) -> None:
        order.append(f"handler:{turn.text}")

    async with app.test_session() as s:
        await s.join()
        await s.say("hi")
        await s.settle()
        await s.say("boom")
        await s.settle()
    assert order == ["a:hi", "b:hi", "handler:hi", "a:boom", "b:boom"]


async def test_say_reply_and_withdraw_record_what_was_spoken_through_the_dialogue_only() -> None:
    app = _app()
    dialogue = app.add(Dialogue(name="voice"))
    trace = MemorySink()
    app.trace.add(trace)

    @app.on_turn
    async def handle(turn: Turn, p: Participant) -> None:
        if turn.text == "say":
            await dialogue.say(p, "Kettle on.", source="kettle")
            assert await dialogue.withdraw(p, "kettle") is True  # dropped from the queue, kept in the log
        elif turn.text == "stream":
            assert await dialogue.reply(p, _parts("It is ", "ten.")) == "It is ten."
        elif turn.text == "blank":
            await dialogue.reply(p, "")
            await dialogue.reply(p, _parts(" ", ""))
        elif turn.text == "direct":
            await p.say("Not recorded.")
            await p.reply("Nor this.")

    async with app.test_session() as s:
        await s.join()
        for text in ("say", "stream", "blank", "direct"):
            await s.say(text)
            await s.settle()
        entries = dialogue.log.entries(s.participant())
    assert entries == [
        "user: say",
        "assistant: Kettle on.",
        "user: stream",
        "assistant: It is ten.",
        "user: blank",
        "user: direct",
    ]
    outputs = [(e.name, e.data["how"], e.data["text"]) for e in trace.events(kind="output")]
    assert outputs == [
        ("voice", "say", "Kettle on."),
        ("voice", "reply", ""),
        ("voice", "reply_end", "It is ten."),
        ("voice", "reply", ""),  # a blank stream is traced as Voice traced it, but not recorded
        ("voice", "reply_end", " "),
    ]
