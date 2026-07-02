from dataclasses import dataclass
import random


@dataclass
class Hex:
    value: int  #  0 <= val < 16


@dataclass
class HexList:
    data: list[Hex]


class ISymbol:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def sample(self) -> Hex:
        raise NotImplementedError


class ConstantSymbol(ISymbol):
    def __init__(self, rng: random.Random, constant: Hex) -> None:
        super().__init__(rng)
        self.constant = constant

    def sample(self) -> Hex:
        return self.constant


class UniformChoiceSymbols(ISymbol):
    def __init__(self, rng: random.Random, choices: list[Hex]) -> None:
        super().__init__(rng)
        self.choices = choices

    def sample(self) -> Hex:
        return self.rng.choice(self.choices)


class Rule:
    def __init__(self, symbols: list[ISymbol]) -> None:
        self.symbols = symbols

    def sample(self) -> HexList:
        return HexList(data=[s.sample() for s in self.symbols])


class DataGenerator:
    def __init__(self, seed=123) -> None:
        self.seed = seed
        rng = random.Random(seed)
        a_or_b = UniformChoiceSymbols(rng, choices=[Hex(0xA), Hex(0xB)])
        constant_c = ConstantSymbol(rng=rng, constant=Hex(0xC))
        self.rule = Rule([a_or_b, constant_c])

    def sample(self) -> HexList:
        return self.rule.sample()


if __name__ == "__main__":
    gen = DataGenerator(seed=44)
    data = gen.sample()
    print(data)
