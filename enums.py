from enum import IntEnum
from typing import Counter

class Color(IntEnum):
    RED = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4
    WILD = 5

class Type(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    SKIP = 10
    REVERSE = 11
    DRAW2 = 12
    CHANGECOLOR = 13
    DRAW4 = 14

class Direction(IntEnum):
    CLOCKWISE = 1
    COUNTERCLOCKWISE = -1