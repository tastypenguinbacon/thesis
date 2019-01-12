from itertools import chain

import numpy as np


class FocusArea:
    def __init__(self, max_row, max_col):
        self.max_row = max_row
        self.max_col = max_col

    def __contains__(self, item):
        row, col = item
        return row in range(self.max_row) and col in range(self.max_col)


class GameOfLife:
    def __init__(self, **vargs):
        self.max_row = vargs['max_row']
        self.max_col = vargs['max_col']
        if vargs['game_of_life'] is not None:
            self.game_of_life = self.trim(vargs['game_of_life'])
        else:
            self.game_of_life = set()

    def trim(self, points):
        boundaries = FocusArea(self.max_row, self.max_col)
        return set(filter(lambda cell: cell in boundaries, points))

    def numpy_array(self):
        out = np.zeros((self.max_row, self.max_col, 1), dtype=np.int8)
        for row, col in self.game_of_life:
            out[row, col, 0] = 1
        return out

    def count(self):
        return len(self.game_of_life)

    def next(self, cells):
        after_add = set(self.game_of_life) | set(cells)
        cells_and_neighbors = after_add | set(chain(*map(lambda c: GameOfLife.neighbors(c), after_add)))
        next_generation = set()
        for cell in cells_and_neighbors:
            neighbor_count = sum(neigh in after_add for neigh in self.neighbors(cell))
            if neighbor_count == 3 or (neighbor_count == 2 and cell in after_add):
                next_generation.add(cell)
        return GameOfLife(
            max_row=self.max_row,
            max_col=self.max_col,
            game_of_life=self.trim(next_generation)
        )

    @staticmethod
    def neighbors(cell):
        row, col = cell
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if not (i == 0 and j == 0):
                    yield row + i, col + j

    def print(self):
        for row in range(self.max_row):
            for col in range(self.max_col):
                if (row, col) in self.game_of_life:
                    print('#', end='')
                else:
                    print(' ', end='')
            print()


