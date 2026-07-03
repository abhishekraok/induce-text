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
    symbols: list[int | BinaryHexChoice]

    def sample(self, choices: Iterator[bool]):
        results = []
        for s in self.symbols:
            if isinstance(s, BinaryHexChoice):
                results.append(s.choose(next(choices)))
            else:
                results.append(s)
        return results


def bitstream(rng: random.Random):
    while True:
        yield rng.choice([True, False])


class DataGenerator:
    def __init__(self, seed: int, rules: list[Rule]) -> None:
        self.seed = seed
        self.rules = rules
        self.rng = random.Random(seed)

    def sample(self) -> list[int]:
        rule = self.rng.choice(self.rules)
        return rule.sample(bitstream(self.rng))


if __name__ == "__main__":
    rule = Rule(
        symbols=[0x0, 0x0, BinaryHexChoice(a=1, b=2), 0x0, BinaryHexChoice(a=3, b=4)]
    )
    for seed in range(4):
        gen = DataGenerator(seed=seed, rules=[rule])
        data = gen.sample()
        print(data)
