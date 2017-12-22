from itertools import chain
import numpy as np


def neighbors(cell):
    i, j = cell
    for y in (-1, 0, 1):
        for x in (-1, 0, 1):
            if not (x == 0 and y == 0):
                yield i + y, j + x


class FocusArea:
    def __init__(self, max_row, max_col):
        self.max_row = max_row
        self.max_col = max_col

    def __contains__(self, item):
        row, col = item
        return row in range(self.max_row) and col in range(self.max_col)


class GameOfLife:
    def __init__(self, focus_area, board=None):
        self.focus_area = focus_area
        self.board = set() if board is None else set(board)

    def next(self):
        next_generation = set()
        cells_to_consider = self.board | set(chain(*map(neighbors, self.board)))
        for cell in cells_to_consider:
            neighbor_count = sum(neigh in self.board for neigh in neighbors(cell))
            if neighbor_count == 3 or (neighbor_count == 2 and cell in self.board):
                if cell in self.focus_area:
                    next_generation.add(cell)
        return GameOfLife(self.focus_area, next_generation)

    def add(self, *cells):
        new_game = self.board | set(cells)
        new_game = set(filter(lambda x: x in self.focus_area, new_game))
        return GameOfLife(self.focus_area, new_game)

    def to_numpy_array(self):
        max_row = self.focus_area.max_row
        max_col = self.focus_area.max_col
        out = np.zeros((max_row, max_col, 1), dtype=np.float32)
        for index in self.board:
            row, col = index
            row, col = int(row), int(col)
            out[row, col, 0] = 1
        return out

    def __len__(self):
        return len(self.board)

    def __str__(self):
        lines = []
        for i in range(self.focus_area.max_row):
            line = ''.join("#" if (i, j) in self.board else " " for j in range(self.focus_area.max_col))
            lines.append(line)
        return '\n'.join(lines) + '\n'


class MultiInputGol:
    def __init__(self, gol, max_cnt, cnt=0):
        self.gol = gol
        self.max_cnt = max_cnt
        self.cnt = cnt

    def next(self, print_out=False):
        cnt = self.cnt + 1
        if cnt >= self.max_cnt:
            if print_out: print(str(self))
            return MultiInputGol(self.gol.next(), self.max_cnt)
        else:
            return MultiInputGol(self.gol, self.max_cnt, cnt)

    def add(self, *cells):
        return MultiInputGol(self.gol.add(*cells), self.max_cnt, self.cnt)

    def to_numpy_array(self):
        return self.gol.to_numpy_array() * (self.cnt + 1) / self.max_cnt

    def __len__(self):
        return len(self.gol)

    def __str__(self):
        return str(self.gol)
