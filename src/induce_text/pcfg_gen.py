from dataclasses import dataclass

type HexList = list[int]  # List of Hexadecimals


@dataclass
class HexListData:
    data: HexList


class DataGenerator:
    def __init__(self, seed=123) -> None:
        self.seed = seed

    def sample(self) -> HexListData:
        return HexListData(data=[0x5, 0xA])
