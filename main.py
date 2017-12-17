from tkinter import Tk

from game_of_life import GameOfLife, FocusArea
import numpy as np
import matplotlib.pyplot as plt

from neural_network import neural_net, decode, random_board

from gui import GameOfLifeBoard, Dim

width, height = 16, 16
focus_area = FocusArea(max_col=width, max_row=height)

brd = GameOfLife(FocusArea(width, height), random_board())

for i in range(height):
    brd = brd.add((i, i))
    brd = brd.add((height - i - 1, i))
    brd = brd.add((i, height // 2))
    brd = brd.add((i, height // 2 - 1))
    brd = brd.add((width // 2, i))
    brd = brd.add((width // 2 - 1, i))

GameOfLifeBoard.pixels_per_box = 10
GameOfLifeBoard.inter_frame_time = 250



class Mutator:
    def __call__(self, brd):
        action = neural_net.predict(np.array(brd.to_numpy_array()))[0]
        return brd.add(decode(action))



root = Tk()
root.title('Conway\'s game of Life')

canv = GameOfLifeBoard(root, Dim(height, width), brd)  # , Mutator(width, height))

canv.tick()

root.mainloop()

plt.plot(canv.cells)
plt.grid(True)
plt.axis([0, len(canv.cells) - 1, 0, max(canv.cells)])
plt.show()
