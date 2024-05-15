import random

WIDTH = 3
HEIGHT = 3


class Board:
    def __init__(self):
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.player = 1
        self.moves = []

    def show_board(self):
        for i in range(HEIGHT):
            str_row = ""
            for j in range(WIDTH):
                str_row += f"{self.board[i][j]}"
            print(str_row)

    def get_moves(self):
        moves = []
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if (self.board[i][j] == 0):
                    moves.append((i, j))
        return moves

    def make_move(self, move: (int, int)):
        self.board[move[0]][move[1]] = self.player
        self.player = self.player % 2 + 1
        self.moves.append(move)

    def make_random_move(self):
        moves = self.get_moves()
        self.make_move(random.choice(moves))

    def is_winning(self):
        last_player = self.player % 2 + 1
        board = self.board
        # top left -> bottom right
        if board[0][0] == board[1][1] == board[2][2] == last_player:
            return True
        # top right -> bottom left
        if board[0][2] == board[1][1] == board[2][0] == last_player:
            return True
        # verticals
        if board[0][0] == board[1][0] == board[2][0] == last_player:
            return True
        if board[0][1] == board[1][1] == board[2][1] == last_player:
            return True
        if board[0][2] == board[1][2] == board[2][2] == last_player:
            return True
        # horizontals
        if board[0][0] == board[0][1] == board[0][2] == last_player:
            return True
        if board[1][0] == board[1][1] == board[1][2] == last_player:
            return True
        if board[2][0] == board[2][1] == board[2][2] == last_player:
            return True
        return False


if __name__ == "__main__":
    board = Board()
    for _ in range(9):
        board.make_random_move()
        board.show_board()
        input()
