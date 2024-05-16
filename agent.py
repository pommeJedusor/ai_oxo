import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense
import random

import oxo_game as oxo

MUTATION_RATE = 0.01
POPULATION_SIZE = 100
LOOSE_BY_INVALID_MOVE = 0
LOOSE_ALIGNMENT = 1
DRAW = 2
WIN = 3
TEST_SIZE = 10


def flat(array):
    final_array = []
    for arr in array:
        final_array.extend(arr)
    return final_array


# input struct
# 27 booleans (1 == True or 0 == False) input
# 1. is square(index 0) empty
# 2. is square(index 0) occupied by player 1
# 3. is square(index 0) occupied by player 2
# 4. is square(index 1) empty
# 5. is square(index 1) occupied by player 1
# 6. is square(index 1) occupied by player 2
# ...
# 28. 0 if ai is player 1, 1 if ai is player 2
def trad_input(game):
    inputs = []
    for square in flat(game.board):
        if square == 0:
            inputs.extend([1, 0, 0])
        elif square == 1:
            inputs.extend([0, 1, 0])
        elif square == 2:
            inputs.extend([0, 0, 1])
    inputs.append(game.player - 1)
    return np.array([inputs])


def trad_output(output):
    max_value = 0
    best_index = 0
    for i in range(len(output[0])):
        value = output[0][i]
        if value > max_value:
            max_value = value
            best_index = i
    move = (best_index // 3, best_index % 3)
    return move


def get_model_move(model, input):
    output = model.predict(input)
    return trad_output(output)


def model_play_model(model_1, model_2):
    game = oxo.Board()
    while True:
        # first model's turn
        model_1_move = get_model_move(model_1, trad_input(game))
        if game.board[model_1_move[0]][model_1_move[1]]:
            return model_2

        game.make_move(model_1_move)

        if game.is_winning():
            return model_1
        if (len(game.moves) >= 9):
            return random.choice([model_1, model_2])

        # second model's turn
        model_2_move = get_model_move(model_2, trad_input(game))
        if game.board[model_2_move[0]][model_2_move[1]]:
            return model_1

        game.make_move(model_2_move)

        if game.is_winning():
            return model_2
        if (len(game.moves) >= 9):
            return random.choice([model_1, model_2])


def model_play_game(model):
    is_first_player = True if random.randint(0, 1) else False
    game = oxo.Board()
    if not is_first_player:
        game.make_random_move()

    while True:
        model_move = get_model_move(model, trad_input(game))
        if game.board[model_move[0]][model_move[1]]:
            return LOOSE_BY_INVALID_MOVE

        game.make_move(model_move)

        if game.is_winning():
            return WIN
        if (len(game.moves) >= 9):
            return DRAW

        game.make_random_move()

        if game.is_winning():
            return LOOSE_ALIGNMENT
        if (len(game.moves) >= 9):
            return DRAW


def save_model(model, filename="model2.keras"):
    model.save(filename)


def get_fitness_value(model):
    total_score = 0
    for _ in range(TEST_SIZE):
        result = model_play_game(model)
        if result == LOOSE_BY_INVALID_MOVE:
            total_score -= 3
        if result == LOOSE_ALIGNMENT:
            total_score -= 1
        if result == DRAW:
            pass
        if result == WIN:
            total_score += 1

    return total_score


def create_model():
    model = Sequential()
    model.add(Dense(50, input_dim=28, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    return model


def get_best_brains(population, deepmax=1, deep=0):
    best_brains = []
    while len(population) > 1:
        # remove random model into model_1
        model_1_index = random.randint(0, len(population) - 1)
        model_1 = population[model_1_index]
        del population[model_1_index]
        # remove random model into model_2
        model_2_index = random.randint(0, len(population) - 1)
        model_2 = population[model_2_index]
        del population[model_2_index]
        # make them fight against each other and keep the winning
        winner = model_play_model(model_1, model_2)
        best_brains.append(winner)
    if deep >= deepmax:
        return best_brains
    else:
        return get_best_brains(best_brains, deepmax, deep+1)


def clone_models(models):
    clones = []
    for model in models:
        clone = clone_model(model)
        clone.set_weights(model.get_weights())
        clones.append(clone)
    return clones


def mutate_model(model, mutation_rate):
    weights = model.get_weights()
    for i in range(len(weights)):
        shape = weights[i].shape
        mutation = np.random.standard_normal(size=shape)
        mask = np.random.binomial(1, mutation_rate, size=shape)
        weights[i] += mutation * mask
    model.set_weights(weights)
    return model


def mutate_models(models, mutate_rate):
    for i in range(len(models)):
        models[i] = mutate_model(models[i], mutate_rate)
    return models


def genetic_algorithm(population, num_gene=300):
    for i in range(num_gene):
        print(f"ai_oxo: -- generation number {i}")
        best_brains = get_best_brains(population)
        clones = clone_models(best_brains)
        mutated_1 = mutate_models(best_brains, MUTATION_RATE)
        mutated_2 = mutate_models(clones, MUTATION_RATE)
        clones = clone_models(best_brains)
        mutated_3 = mutate_models(clones, MUTATION_RATE)
        clones = clone_models(best_brains)
        mutated_4 = mutate_models(clones, MUTATION_RATE)
        save_model(random.choice(best_brains))
        mutated_1.extend(mutated_2)
        mutated_1.extend(mutated_3)
        mutated_1.extend(mutated_4)
        population = mutated_1
    print("ai_oxo: finished executing genetic_algorithm")


if __name__ == "__main__":
    population = [create_model() for _ in range(POPULATION_SIZE)]
    genetic_algorithm(population)
