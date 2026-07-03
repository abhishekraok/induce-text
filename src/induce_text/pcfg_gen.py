from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Iterator, Protocol


class ChoiceSource(Protocol):
    def choice(self) -> bool: ...


class RecordingChoice:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.choices: list[bool] = []

    def choice(self) -> bool:
        self.choices.append(bool(self.rng.getrandbits(1)))
        return self.choices[-1]

    @property
    def count(self):
        return len(self.choices)

    def __str__(self) -> str:
        return "".join(str(int(c)) for c in self.choices)


class ReplayChoice:

    def __init__(self, choices: list[bool]):
        self.choices = choices
        self.index = 0

    def choice(self) -> bool:
        if self.index >= len(self.choices):
            raise ValueError("choice stream exhausted")
        value = self.choices[self.index]
        self.index += 1
        return value


@dataclass
class Rule:
    symbols: list[str | list[str]]


def sample(
    rule: Rule, env: dict[str, int | Rule], choicesource: ChoiceSource
) -> list[int]:
    result = []
    for s in rule.symbols:
        if isinstance(s, list):
            choice = choicesource.choice()
            actual_s = s[0] if choice else s[1]
        else:
            actual_s = s
        if actual_s not in env:
            raise ValueError(f"Symbol {actual_s} not found in environment {env}")
        val = env[actual_s]
        if isinstance(val, Rule):
            result.extend(sample(val, env, choicesource))
        else:
            result.append(val)
    return result


if __name__ == "__main__":
    x = Rule(symbols=["a", "b", ["y", "d"], "e", "f", ["x", "a"]])
    y = Rule(symbols=["a", "b", ["c", "d"]])

    env = {"a": 10, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "x": x, "y": y}
    for seed in range(4):
        cs = RecordingChoice(seed=seed)
        data = sample(rule=x, env=env, choicesource=cs)
        hex_str = "".join([f"{x:x}" for x in data])
        print(f"Raw data {hex_str}, choices {cs}")
