#!/usr/bin/env python
"""banter.py -- the engine's voice (Kaveh 2026-08-13: "plant an LLM in the middle to handle
the emojis and text responses ... it could be very dumb and small and fast since it's not
critical work" + "I need something free").

DESIGN CONTRACT: the DECISION is mechanical and stays in the server (surprise bits, gotcha
memory, prep state -- engine ground truth). Banter only PHRASES an already-decided event,
so a hallucinated line is harmless: it can never claim a gotcha that did not happen.

Backends (all free):
    TemplateBanter   zero-cost rotating templates; the default and the fallback
    OllamaBanter     any local ollama model via /api/generate; hard timeout, template
                     fallback on any failure. A one-line quip is ~30 tokens.

speak(event) -> {"emoji": str, "text": str}; event = {"kind", "san", "bits", "player",
"moves" (recent san line), ...}. The emoji is decided by KIND (mechanical), never by the
LLM -- the bubble's big emoji is part of the contract, the words are the creative part.
"""
from __future__ import annotations

import json
import urllib.request

EMOJI = {"surprised": "\U0001F62E", "gotcha": "\U0001F61D"}


class TemplateBanter:
    """rotating canned lines; also the fallback wording for every other backend."""

    T = {"surprised": ["{san}?! I did not see that coming ({bits:.1f} bits)",
                       "{san}?? that was nowhere in my book ({bits:.1f} bits)",
                       "wait -- {san}? noted. I will remember that.",
                       "{san}! bold. {bits:.1f} bits of bold."],
         "gotcha": ["{san}? You got me with that once. I did my homework.",
                    "{san} again? I studied that line while you were away.",
                    "not this time. {san} is in my notebook now."]}

    def __init__(self):
        self._n = {}

    def speak(self, event):
        kind = event.get("kind", "surprised")
        i = self._n.get(kind, 0)
        self._n[kind] = i + 1
        lines = self.T.get(kind) or self.T["surprised"]
        return {"emoji": EMOJI.get(kind, "\U0001F916"),
                "text": lines[i % len(lines)].format(**{"san": event.get("san", "?"),
                                                        "bits": float(event.get("bits", 0))})}


class OllamaBanter:
    """local ollama model. Free, offline, and allowed to be dumb -- on ANY failure or
    slow reply the template speaks instead, so play never blocks on the LLM."""

    SYS = ("You are catspace, a chess engine with a dry, playful personality. "
           "Reply with EXACTLY one emoji that captures your reaction, then a space, "
           "then ONE short line of words (max 90 characters), no quotes, no "
           "explanations. Refer to the move played. Never invent facts beyond the "
           "event given. Example reply: \U0001F624 g4? Bold of you to assume I sleep.")

    PROMPT = {"surprised": ("Your opponent {player} just played {san}, a move you gave "
                            "almost no probability ({bits:.1f} bits of surprise, higher = "
                            "more shocking). React in one line. Mention no move other "
                            "than {san}."),
              "gotcha": ("Your opponent {player} played {san}, which SURPRISED you in a "
                         "past game -- but you studied it between sessions and this time "
                         "you replied instantly with a prepared line. Gloat in one line. "
                         "Mention no move other than {san}.")}

    def __init__(self, model="llama3.1:8b", url="http://127.0.0.1:11434",
                 timeout=4.0):
        self.model, self.url, self.timeout = model, url, timeout
        self._fallback = TemplateBanter()

    def warm(self):
        """load the model into ollama's memory (first generate pays ~8s; keep_alive 30m
        makes in-game quips ~1-3s). Call from a background thread at server start."""
        try:
            req = urllib.request.Request(
                self.url + "/api/generate",
                data=json.dumps({"model": self.model, "prompt": "ready?", "stream": False,
                                 "keep_alive": "30m",
                                 "options": {"num_predict": 2}}).encode(),
                headers={"content-type": "application/json"})
            urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            pass

    def speak(self, event):
        kind = event.get("kind", "surprised")
        try:
            prompt = self.PROMPT.get(kind, self.PROMPT["surprised"]).format(
                player=event.get("player") or "the human",
                san=event.get("san", "?"), bits=float(event.get("bits", 0)))
            if event.get("moves"):
                prompt += " Recent moves: " + event["moves"] + "."
            req = urllib.request.Request(
                self.url + "/api/generate",
                data=json.dumps({"model": self.model, "prompt": prompt,
                                 "system": self.SYS, "stream": False,
                                 "keep_alive": "30m",
                                 "options": {"num_predict": 30,
                                             "temperature": 0.9}}).encode(),
                headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                txt = json.loads(r.read()).get("response", "").strip().strip('"')
            txt = txt.splitlines()[0][:120] if txt else ""
            if not txt:
                raise ValueError("empty reply")
            # the LLM PICKS the emoji (Kaveh 2026-08-13): a leading non-ascii token is
            # the reaction; anything else falls back to the kind's emoji
            emoji = EMOJI.get(kind, "\U0001F916")
            head, _, rest = txt.partition(" ")
            if head and rest and not head[0].isascii():
                emoji, txt = head, rest.strip()
            # format discipline: the line must be WORDS -- a reply that is empty, tiny,
            # or emoji-soup falls back to the template (small models drift)
            if len(txt) < 4 or not any(c.isalpha() for c in txt):
                raise ValueError("no words")
            return {"emoji": emoji, "text": txt}
        except Exception:
            return self._fallback.speak(event)


def make_banter(spec: str):
    """'template' | 'ollama' | 'ollama:<model>' -> a speaker."""
    if spec and spec.startswith("ollama"):
        _, _, model = spec.partition(":")
        return OllamaBanter(model=model or "llama3.1:8b")
    return TemplateBanter()


def _tests():
    ok = True
    tb = TemplateBanter()
    a = tb.speak({"kind": "surprised", "san": "g4", "bits": 4.3})
    b = tb.speak({"kind": "surprised", "san": "h4", "bits": 3.1})
    ok &= a["emoji"] == EMOJI["surprised"] and "g4" in a["text"] and a["text"] != b["text"]
    g = tb.speak({"kind": "gotcha", "san": "g4", "bits": 0.2})
    ok &= g["emoji"] == EMOJI["gotcha"] and "g4" in g["text"]
    # ollama backend NEVER raises: dead endpoint -> template fallback, fast
    ob = OllamaBanter(model="nope", url="http://127.0.0.1:9", timeout=0.3)
    f = ob.speak({"kind": "surprised", "san": "e4", "bits": 5.0})
    ok &= "e4" in f["text"] and f["emoji"] == EMOJI["surprised"]
    print(f"[banter] templates rotate, kinds keep their emoji, dead-LLM falls back  "
          f"{'OK' if ok else 'FAIL'}")
    print("ALL BANTER TESTS PASSED" if ok else "TESTS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _tests()
