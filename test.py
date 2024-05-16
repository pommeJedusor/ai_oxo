import numpy as np
import random
from tensorflow.keras.models import load_model
import oxo_game as oxo
import agent


def flat(array):
    final_array = []
    for arr in array:
        final_array.extend(arr)
    return np.array(final_array)


def get_user_move(game):
    user_move = input("enter your move 1 == top left, 3 == top right and 9 == bottom right: ")
    try:
        move = int(user_move)
        move -= 1
        if move > 8 or move < 0:
            print("move non valide")
        elif game.board[move // 3][move % 3]:
            print("move non valide")
        else:
            return (move // 3, move % 3)
    except:
        print("move non valide")
    return get_user_move(game)


def test_ai(model):
    game = oxo.Board()
    is_ai_first = True if random.randint(0, 1) else False
    if not is_ai_first:
        user_move = get_user_move(game)
        game.make_move(user_move)
        game.show_board()
    while True:
        model_move = agent.get_model_move(model, agent.trad_input(game))
        print(f"ai play {model_move}")
        if game.board[model_move[0]][model_move[1]]:
            print(f"ia tried {model_move} and lost")
            return
        game.make_move(model_move)
        game.show_board()
        if game.is_winning():
            print("ai won")
            return
        if len(game.moves) >= 9:
            print("draw")
            return

        user_move = get_user_move(game)
        game.make_move(user_move)
        game.show_board()
        if game.is_winning():
            print("you won")
            return
        if len(game.moves) >= 9:
            print("draw")
            return


if __name__ == "__main__":
    # load model
    model = load_model('model.keras')
    test_ai(model)
