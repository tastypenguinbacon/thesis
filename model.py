import keras


def ranged_score(lower_bound, upper_bound):
    def score(game_board):
        count = len(game_board)
        return (count - lower_bound) * (count - upper_bound)

    return score
