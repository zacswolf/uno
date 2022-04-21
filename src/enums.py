from enum import IntEnum


class Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3
    WILD = 4


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
