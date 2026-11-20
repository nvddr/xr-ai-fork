# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hello voice: the smallest spoken assistant.

SHAPE: one component answers every turn. There is no router: `app.on_turn`
names the one handler, and the conversation's `answer` is that handler.

  user speaks -> speech to text -> app.on_turn -> conversation.answer
              -> p.reply(..., to=dialogue) -> text to speech

The dialogue component speaks and keeps what was said (`dialogue.log`, a
context log), and the conversation reads it, so the assistant remembers
earlier turns.
"""

from pathlib import Path

from xr_ai_app import App, Hub, PerConnection
from xr_ai_kit import Conversation, Dialogue

HERE = Path(__file__).resolve().parent

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
app.add(Hub())  # the device connection: who is here

# Model roles: models.json says which model fills each one.
answer = app.role("answer_lightweight")

# Speech out, and the dialogue log. Which speech models, and when to listen,
# is app.yaml's `voice:` section. `summarize` compacts the log when it is full.
dialogue = app.add(Dialogue(summarize=answer, budget_bytes=6_000))

conversation = app.add(
    Conversation(
        role=answer,
        output=dialogue,
        instructions="You are a friendly assistant. Answer in one or two "
        "short spoken sentences.",
        context=[dialogue.log],
    )
)

# Every turn goes to the conversation. (Swap in a Router when there is more
# than one thing to send turns to: see 04-notes-mode.)
app.on_turn(conversation.answer)

if __name__ == "__main__":
    app.run()
