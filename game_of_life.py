from itertools import chain


def neighbors(cell):
    i, j = cell[0], cell[1]
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            if not (x == 0 and y == 0):
                yield i + x, j + y


class Box:
    def __init__(self, max_x, max_y):
        self.max_x = max_x
        self.max_y = max_y

    def __contains__(self, item):
        x, y = item[0], item[1]
        return x in range(self.max_x) and y in range(self.max_y)


class GameBoard:
    def __init__(self, box, board=None):
        self.box = box
        self.board = set() if board is None else board

    def __iter__(self):
        return self

    def __next__(self):
        next_generation = set()
        recalc = self.board | set(chain(*map(neighbors, self.board)))
        for cell in recalc:
            count = sum((neigh in self.board) for neigh in neighbors(cell))
            if count == 3 or (count == 2 and cell in self.board):
                if cell in self.box:
                    next_generation.add(cell)
        self.board = next_generation

    def add(self, cell):
        if cell in self.box:
            self.board.add(cell)

    def __len__(self):
        return sum((cell in self.board ) for cell in self.board)

    def __str__(self):
        limits = self.box
        result = []
        for y in range(limits.max_y):
            for x in range(limits.max_x):
                if (x, y) in self.board:
                    result.append('#')
                else:
                    result.append(' ')
            result.append('\n')
        return ''.join(result)
