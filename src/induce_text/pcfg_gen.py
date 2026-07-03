from __future__ import annotations
from dataclasses import dataclass
import random
from typing import NamedTuple, Iterator


class BinaryHexChoice(NamedTuple):
    """Stores two hexadecimal values."""

    a: int
    b: int

    def choose(self, choice: bool) -> int:
        return self.a if choice else self.b


@dataclass
class Rule:
    symbols: list[int | BinaryHexChoice | Rule]

    def sample(self, choices: Iterator[bool]) -> list[int]:
        results = []
        for s in self.symbols:
            if isinstance(s, BinaryHexChoice):
                results.append(s.choose(next(choices)))
            elif isinstance(s, Rule):
                results.extend(s.sample(choices))
            else:
                results.append(s)
        return results


class ChoiceMaker:
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
        self.count = 0

    def bitstream(self):
        while True:
            self.count += 1
            yield self.rng.choice([True, False])


class DataGenerator:
    def __init__(self, rule: Rule) -> None:
        self.rule = rule

    def sample(self, choicemaker: ChoiceMaker) -> list[int]:
        return self.rule.sample(choicemaker.bitstream())


if __name__ == "__main__":
    rule_5_then_a_or_b = Rule(symbols=[5, BinaryHexChoice(a=0xA, b=0xB)])
    rule = Rule(symbols=[0x0, 0x0, rule_5_then_a_or_b, 0x0, BinaryHexChoice(a=3, b=4)])
    gen = DataGenerator(rule=rule)
    for seed in range(4):
        choicemaker = ChoiceMaker(seed=seed)
        data = gen.sample(choicemaker)
        print(data)
