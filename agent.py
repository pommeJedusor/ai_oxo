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


def get_model_move(model, input):
    output = model.predict(input)
    max_value = 0
    best_index = 0
    for i in range(len(output[0])):
        value = output[0][i]
        if value > max_value:
            max_value = value
            best_index = i
    move = (best_index // 3, best_index % 3)
    return move


def model_play_game(model):
    is_first_player = True if random.randint(0,1) else False
    game = oxo.Board()
    if not is_first_player:
        game.make_random_move()
    
    while True:
        model_move = get_model_move(model, np.array([flat(game.board)]))
        if game.board[model_move[0]][model_move[1]]:
            return LOOSE_BY_INVALID_MOVE

        game.make_move(model_move)

        if game.is_winning():
            return WIN
        if (len(game.moves)>=9):
            return DRAW
        
        game.make_random_move()

        if game.is_winning():
            return LOOSE_ALIGNMENT
        if (len(game.moves)>=9):
            return DRAW


def save_model(model, filename="model.keras"):
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
    model.add(Dense(50, input_dim=9, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    return model


def get_best_brains(population, fitness_values):
    brains = []
    for i in range(len(population)):
        brain = {"brain": population[i], "value": fitness_values[i]}
        brains.append(brain)
    brains.sort(key=lambda x: x["value"], reverse=True)
    brains = [brain["brain"] for brain in brains]
    return brains[0:len(brains)//2]


def get_best_brain(population, fitness_values):
    max_value = max(fitness_values)
    index = fitness_values.index(max_value)
    return population[index]


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


def genetic_algorithm(population, num_gene=100):
    for i in range(num_gene):
        print(f"-- generation number {i}")
        fitness_values = [get_fitness_value(brain) for brain in population]
        best_brains = get_best_brains(population, fitness_values)
        clones = clone_models(best_brains)
        mutated = mutate_models(clones, MUTATION_RATE)
        best_brains.extend(mutated)
        save_model(get_best_brain(population, fitness_values))
        population = best_brains
    print("finished executing genetic_algorithm")


if __name__ == "__main__":
    population = [create_model() for _ in range(POPULATION_SIZE)]
    genetic_algorithm(population)
