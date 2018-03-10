import numpy as np

from agents import ActorCritic
from game_of_life import FocusArea, GameOfLife

width, height = 8, 8
focus_area = FocusArea(max_col=width, max_row=height)
number_of_epochs = 4000
game_iterations = 400
exploration_rate = 1
gamma = 0.9


class Reward:
    def __init__(self, min_cells, max_cells):
        self.min_cells = min_cells
        self.max_cells = max_cells

    def __call__(self, nxt, action):
        mid = (self.min_cells + self.max_cells) / 2
        dif = (self.max_cells - self.min_cells) / 2
        cnt = len(nxt)
        outside = 0
        for row, col in action:
            if not row in range(height) or col not in range(width):
                outside -= 10
        return -np.abs(cnt - mid) + dif + outside


class BinaryReward:
    def __init__(self, min_cells, max_cells):
        self.min_cells = min_cells
        self.max_cells = max_cells

    def __call__(self, brd, nxt, bad):
        if self.min_cells <= len(nxt) <= self.max_cells:
            return 1
        else:
            return -0.1


def random_board(size=None):
    how_many = np.random.randint(0, width * height) if size is None else size
    cols = np.random.randint(0, width, how_many)
    rows = np.random.randint(0, height, how_many)
    return zip(rows, cols)


def deep_q_learning():
    reward = Reward(12, 16)
    name = 'cudo_binary.be'
    # nnet = DQN(params={
    #     'input_size': (height, width),
    #     'exploration_probability': exploration_rate,
    #     'batch_size': 2 ** 8,
    #     'epochs': 1,
    #     'learning_rate': gamma,
    #     'max_mem': 2 ** 15
    # })

    nnet = ActorCritic(params={
        'width': width,
        'height': height,
        'gamma': gamma,
        'batch_size': 512,
        'exploration_rate': exploration_rate,
        'cells_to_add': 3
    })

    nnet.load(name)
    global exploration_rate
    for c in range(number_of_epochs):
        cnt, bad_cnt, cnt_rand = 0, 0, [0] * 20
        board = GameOfLife(focus_area, random_board())
        bad = board
        rand_board = [board] * 20
        nnet.params['exploration_rate'] *= 0.95
        if nnet.params['exploration_rate'] < 10 ** -3:
            nnet.params['exploration_rate'] = 1
        for j in range(game_iterations):
            if 12 <= len(board) <= 16:
                cnt += 1
            if 12 <= len(bad) <= 16:
                bad_cnt += 1
            bad = bad.next()
            for i in range(20):
                if 12 <= len(rand_board[i]) <= 16:
                    cnt_rand[i] += 1
                rand_board[i] = rand_board[i].add(*nnet.propose_action(board, True)).next()
            action = nnet.propose_action(board)
            next_board = board.add(*action).next()
            r = reward(next_board, action)
            if c * game_iterations + j > 512 * 4:
                print(board)
                print((c, j), nnet.params['exploration_rate'], action, r, len(nnet.memory), len(board), '->',
                      len(next_board))
                print(cnt, bad_cnt, cnt_rand)

            nnet.remember((board, action, r, next_board))
            board = next_board
            if j % 10 == 0:
                nnet.learn()
        nnet.save(name)


if __name__ == '__main__':
    deep_q_learning()
