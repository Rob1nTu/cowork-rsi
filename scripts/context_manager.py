"""Bounded context management for long-running tool-call agents.

The full trajectory remains the durable evidence. When the prompt approaches the
model window, the API view keeps the task contract and a recent tool-call tail,
while writing a compact checkpoint into the workspace for on-demand retrieval.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ContextBudgetExceeded(ValueError):
    pass


class ContextWindowManager:
    def __init__(
        self,
        *,
        context_limit: int,
        max_output_tokens: int,
        workspace: Path,
        audit_path: Path,
        input_safety_tokens: int = 1024,
        recent_messages: int = 12,
    ) -> None:
        self.context_limit = context_limit
        self.max_output_tokens = max_output_tokens
        self.input_budget = context_limit - max_output_tokens - input_safety_tokens
        self.workspace = workspace
        self.audit_path = audit_path
        self.recent_messages = recent_messages
        self.compactions = 0
        self.last_estimate = 0
        self.last_kept = 0
        self.last_dropped = 0

    def _message_text(self, message: dict[str, Any]) -> str:
        parts = [str(message.get("role", "")), str(message.get("content") or "")]
        if message.get("reasoning"):
            parts.append(str(message["reasoning"]))
        if message.get("reasoning_content"):
            parts.append(str(message["reasoning_content"]))
        if message.get("tool_calls"):
            parts.append(json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True))
        return "\n".join(parts)

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        text = "\n".join(self._message_text(message) for message in messages)
        # Conservative character heuristic: CJK commonly maps close to one token
        # per character; ASCII prose is denser, while code/tool schemas need margin.
        estimated = 0.0
        for character in text:
            codepoint = ord(character)
            if character.isascii():
                estimated += 0.5 if character.isalnum() else 0.35
            elif 0x2E80 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF:
                estimated += 1.0
            else:
                estimated += 1.0
        return int(estimated) + 4 * len(messages) + 512  # tool schema and chat template margin

    def _complete_recent_tail(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tail = messages[-self.recent_messages :]
        while tail and tail[0].get("role") == "tool":
            tail = tail[1:]
        return [self._trim_observation(message) for message in tail]

    def _checkpoint(self, messages: list[dict[str, Any]]) -> Path:
        path = self.workspace / ".cowork_context" / "checkpoint.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = path.parent / "history.jsonl"
        archive.write_text(
            "".join(json.dumps(message, ensure_ascii=False, default=repr) + "\n" for message in messages[2:]),
            encoding="utf-8",
        )
        lines = [
            "# Context checkpoint",
            "",
            "Earlier conversation messages were compacted by the harness.",
            "The full compacted messages, including reasoning and tool results, are in `.cowork_context/history.jsonl`.",
            "Persistent files and outputs remain in `/workspace`; inspect them with bash before acting.",
            "The following is an index of earlier model actions and observations, not a grading answer:",
            "",
        ]
        for index, message in enumerate(messages[2:], start=1):
            text = self._message_text(message).replace("\x00", " ").strip()
            if len(text) > 500:
                text = text[:350] + " ... " + text[-100:]
            lines.append(f"{index}. {message.get('role', '?')}: {text}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _pointer(self, path: Path, dropped: int) -> dict[str, Any]:
        return {
            "role": "user",
            "content": (
                f"Context checkpoint created at `{path.relative_to(self.workspace)}`. "
                f"The harness compacted {dropped} older messages. "
                "Read that file only when earlier actions are needed; current files in /workspace are authoritative."
            ),
        }

    def _trim_observation(self, message: dict[str, Any]) -> dict[str, Any]:
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            return message
        content = message["content"]
        if len(content) <= 6000:
            return message
        trimmed = dict(message)
        trimmed["content"] = content[:4000] + "\n[older observation truncated; inspect checkpoint/workspace]\n" + content[-1500:]
        return trimmed

    def prepare(self, messages: list[dict[str, Any]], *, force: bool = False) -> list[dict[str, Any]]:
        estimate = self.estimate_tokens(messages)
        self.last_estimate = estimate
        if not force and estimate <= self.input_budget:
            self.last_kept = len(messages)
            self.last_dropped = 0
            return messages

        checkpoint = self._checkpoint(messages)
        prefix = messages[:2]
        if force:
            tail = [self._trim_observation(message) for message in messages[-2:]]
            managed = prefix + [self._pointer(checkpoint, len(messages) - len(prefix) - len(tail))] + tail
        else:
            tail = self._complete_recent_tail(messages[2:])
            managed = prefix + [self._pointer(checkpoint, len(messages) - len(prefix) - len(tail))] + tail
        while self.estimate_tokens(managed) > self.input_budget and len(tail) > 2:
            drop = 2 if tail[0].get("role") == "assistant" and tail[1].get("role") == "tool" else 1
            tail = tail[drop:]
            while tail and tail[0].get("role") == "tool":
                tail = tail[1:]
            managed = prefix + [self._pointer(checkpoint, len(messages) - len(prefix) - len(tail))] + tail
        if self.estimate_tokens(managed) > self.input_budget:
            recent = self._complete_recent_tail(messages[-2:])
            managed = prefix + [self._pointer(checkpoint, len(messages) - len(prefix))] + recent
        if self.estimate_tokens(managed) > self.input_budget:
            raise ContextBudgetExceeded(
                f"context remains above input budget: {self.estimate_tokens(managed)} > {self.input_budget}"
            )
        self.compactions += 1
        self.last_kept = len(managed)
        self.last_dropped = len(messages) - len(managed)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "timestamp": time.time(),
                "before_messages": len(messages),
                "after_messages": len(managed),
                "before_estimate_tokens": estimate,
                "after_estimate_tokens": self.estimate_tokens(managed),
                "input_budget": self.input_budget,
                "checkpoint": str(checkpoint),
                "forced_after_context_error": force,
            }, ensure_ascii=False) + "\n")
        return managed

    def serialize(self) -> dict[str, int]:
        return {
            "context_limit": self.context_limit,
            "max_output_tokens": self.max_output_tokens,
            "input_budget": self.input_budget,
            "compactions": self.compactions,
            "last_estimate": self.last_estimate,
            "last_kept": self.last_kept,
            "last_dropped": self.last_dropped,
        }
