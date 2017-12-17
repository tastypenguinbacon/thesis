from collections import namedtuple
from tkinter import *

Dim = namedtuple('Dim', ['height', 'width'])
Coordinate = namedtuple('Coordinate', ['x', 'y'])
Cell = namedtuple('Cell', ['top_left', 'bottom_right'])


def create_cells(box_size, width, height):
    x_cells, y_cells = width // box_size, height // box_size
    result = {}
    for i in range(x_cells):
        for j in range(y_cells):
            top_left = Coordinate(i * box_size + 1, j * box_size + 1)
            bottom_right = Coordinate((i + 1) * box_size,
                                      (j + 1) * box_size)
            result[(i, j)] = Cell(top_left, bottom_right)
    return result


class GameOfLifeBoard(Canvas):
    pixels_per_box = 16
    inter_frame_time = 10

    def __init__(self, parent, dimension, game, mutator=lambda x: x):
        super().__init__(parent, height=dimension.height * GameOfLifeBoard.pixels_per_box,
                         width=dimension.width * GameOfLifeBoard.pixels_per_box)
        self.game = game
        self.grid(row=1, column=1)
        self.cells = create_cells(GameOfLifeBoard.pixels_per_box,
                                  dimension.width * GameOfLifeBoard.pixels_per_box,
                                  dimension.height * GameOfLifeBoard.pixels_per_box)
        self.boxes = self.cells
        self.prev = set()
        self.mutator = mutator
        self.cells = []

    def tick(self):
        self.game = self.game.next()
        cells = len(self.game)
        self.cells.append(cells)
        self.game = self.mutator(self.game)
        self.set_squares(self.game.board)
        self.after(GameOfLifeBoard.inter_frame_time, self.tick)

    def set_squares(self, cells):
        self.delete('all')
        self.prev = set()
        for indices in cells:
            top_left, bottom_right = self.boxes[indices]
            self.prev.add(indices)
            self.create_rectangle(top_left.x, top_left.y,
                                  bottom_right.x, bottom_right.y, fill='green')
