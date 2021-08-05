import numpy as np
from random import random


def init_traders(shape, init_up=0.5):
    """
    Initialise two arrays of spins randomly pointing up or down.
    """
    color_shape = (shape[0], int(shape[1] / 2), shape[2])
    black = np.ones(color_shape)
    white = np.ones(color_shape)

    for color in (black, white):
        for row in range(color_shape[0]):
            for col in range(color_shape[1]):
                for lid in range(color_shape[2]):
                    if random() < init_up:
                        color[row][col][lid] = -1

    return black, white


def precompute_probabilities(reduced_neighbor_coupling, market_coupling):
    """
    Precompute all possible values for the flip-probabilities.
    """
    probabilities = dict()
    for spin in range(-1, 2):
        probabilities[spin] = dict()
        for neighbor_sum in range(-6, 8, 2):
            field = reduced_neighbor_coupling * neighbor_sum - market_coupling * spin
            probabilities[spin][neighbor_sum] = 1 / (1 + np.exp(field))
    return probabilities


def update_strategies(is_black, source, checkerboard_agents, probabilities):
    """
    Update all spins in one array according to the Heatbath dynamic.
    """
    grid_height, grid_width, grid_depth = source.shape
    for row in range(grid_height):
        for col in range(grid_width):
            for lid in range(grid_depth):
                spin = source[row][col][lid]
                lower_neighbor_row = row + 1 if (row + 1 < grid_height) else 0
                upper_neighbor_row = row - 1
                right_neighbor_col = col + 1 if (col + 1 < grid_width) else 0
                left_neighbor_col = col - 1
                front_neighbor_lattice = lid - 1
                back_neighbor_lattice = lid + 1 if (lid + 1 < grid_depth) else 0

                if (not is_black if (lid % 2) else is_black):
                    horizontal_neighbor_col = left_neighbor_col if row % 2 else right_neighbor_col
                else:
                    horizontal_neighbor_col = right_neighbor_col if row % 2 else left_neighbor_col

                neighbor_sum = (
                    checkerboard_agents[upper_neighbor_row][col][lid]
                +   checkerboard_agents[lower_neighbor_row][col][lid]
                +   checkerboard_agents[row][col][lid]
                +   checkerboard_agents[row][horizontal_neighbor_col][lid]
                +   checkerboard_agents[row][col][front_neighbor_lattice]
                +   checkerboard_agents[row][col][back_neighbor_lattice]
                )

                if random() < probabilities[spin][neighbor_sum]:
                    source[row][col][lid] = 1
                else:
                    source[row][col][lid] = -1


def update(black, white, reduced_neighbor_coupling, reduced_alpha):
    """
    Update both arrays.
    """
    number_of_traders = 2 * black.shape[0] * black.shape[1] * black.shape[2]
    global_market = np.sum(black + white)
    market_coupling = reduced_alpha * np.abs(global_market) / number_of_traders
    probabilities = precompute_probabilities(reduced_neighbor_coupling, market_coupling)
    update_strategies(True, black, white, probabilities)
    update_strategies(False, white, black, probabilities)

    return global_market
