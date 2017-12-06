from tkinter import Tk

from game_of_life import GameBoard, Box
from gui import GameOfLife, Dim
import numpy as np
import matplotlib.pyplot as plt

width, height = 128, 128
brd = GameBoard(Box(width, height))

for i in range(height):
    brd.add((i, i))
    brd.add((height - i - 1, i))
    brd.add((i, height // 2))
    brd.add((i, height // 2 - 1))
    brd.add((width // 2, i))
    brd.add((width // 2 - 1, i))

GameOfLife.pixels_per_box = 16


class Mutator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ctr = 0

    def __call__(self, brd):
        for i in range(self.width):
            if np.random.choice([True] + [False] * 178):
                brd.add((self.ctr, i))
            if np.random.choice([True] + [False] * 48):
                brd.add((i, self.ctr))

        self.ctr += 1
        self.ctr %= width


root = Tk()
root.title('Conway\'s game of Life')

canv = GameOfLife(root, Dim(height, width), brd)  # , Mutator(width, height))

canv.tick()

root.mainloop()

plt.plot(canv.cells)
plt.grid(True)
plt.axis([0, len(canv.cells) - 1, 0, max(canv.cells)])
plt.show()
