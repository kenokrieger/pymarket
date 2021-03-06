import numpy as np
from random import random


def init_traders(shape, init_up=0.5):
    """
    Initialise two arrays of spins randomly pointing up or down.

    :param shape: The desired 2 dimensional shape of the array.
    :type shape: tuple
    :param init_up: The percentage of spins pointing up initially. Defaults
        to 0.5.
    :type init_up: float

    :return: The generated arrays.
    :rtype: tuple

    """
    color_shape = (shape[0], int(shape[1] / 2))
    black = np.ones(color_shape, dtype=np.byte)
    white = np.ones(color_shape, dtype=np.byte)

    for color in (black, white):
        for row in range(color_shape[0]):
            for col in range(color_shape[1]):
                if random() < init_up:
                    color[row, col] = -1

    return black, white


def precompute_probabilities(reduced_neighbor_coupling, market_coupling):
    """
    Precompute all possible values for the flip-probabilities.

    :param reduced_neighbor_coupling: The parameter j multiplied by -2 times
        beta.
    :type reduced_neighbor_coupling: float
    :param market_coupling: The parameter alpha multiplied by the absolute
        value of the relative magnetisation.

    :return: The computed probabilities.
    :rtype: `np.ndarray`

    """
    probabilities = np.empty((2, 5), dtype=float)
    for row in range(2):
        spin = 1 if row else -1
        for col in range(5):
            neighbour_sum = -4 + 2 * col
            field = reduced_neighbor_coupling * neighbour_sum - market_coupling * spin
            probabilities[row][col] = 1 / (1 + np.exp(field))
    return probabilities


def compute_neighbour_sum(is_black, source, row, col):
    """
    Calculate the neighbour sum of an individual spin at given position.

    :param is_black: States whether the updated spin is located in the "black"
        or in the "white" array according to the checkerboard algorithm.
    :type is_black: bool
    :param source: The array containing the neighbour spins.
    :type source: `np.ndarray`
    :param row: The row of the spin to compute the neighbour sum of.
    :type row: int
    :param col: The column of the spin to compute the neighbour sum of.
    :type col: int

    :return: The neighbour sum.
    :rtype: int

    """
    grid_height, grid_width = source.shape

    lower_neighbor_row = row + 1 if (row + 1 < grid_height) else 0
    upper_neighbor_row = row - 1
    right_neighbor_col = col + 1 if (col + 1 < grid_width) else 0
    left_neighbor_col = col - 1

    if is_black:
        horizontal_neighbor_col = left_neighbor_col if row % 2 else right_neighbor_col
    else:
        horizontal_neighbor_col = right_neighbor_col if row % 2 else left_neighbor_col

    neighbour_sum = (
        source[upper_neighbor_row, col]
    +   source[lower_neighbor_row, col]
    +   source[row, col]
    +   source[row, horizontal_neighbor_col]
    )
    return int(neighbour_sum)


def update_strategies(is_black, source, checkerboard_agents, probabilities):
    """
    Update all spins in one array according to the Heatbath dynamic.

    :param is_black: States whether the updated spin is located in the "black"
        or in the "white" array according to the checkerboard algorithm.
    :type is_black: bool
    :param source: The array containing the spins to update.
    :type source: `np.ndarray`
    :param checkerboard_agents: The array containing the neighbour spins.
    :type checkerboard_agents: `np.ndarray`
    :param probabilities: The precomputed probabilities for the spin
        orientation.
    :type probabilities: `np.ndarray`

    """
    grid_height, grid_width = source.shape

    for row in range(grid_height):
        for col in range(grid_width):
            neighbour_sum = compute_neighbour_sum(is_black, checkerboard_agents,
                                                  row, col)
            spin_idx = int((source[row, col] + 1) / 2)
            sum_idx = int((neighbour_sum + 4) / 2)
            if random() < probabilities[spin_idx][sum_idx]:
                source[row, col] = 1
            else:
                source[row, col] = -1


def update(black, white, reduced_neighbour_coupling, reduced_alpha):
    """
    Perform a complete lattice update.

    :param black: The array containing the black tiles.
    :type black: `np.ndarray`
    :param white: The array containing the white tiles.
    :type white: `np.ndarray`
    :param reduced_neighbour_coupling: The parameter j multiplied by -2 times
        beta.
    :type reduced_neighbour_coupling: float
    :param reduced_alpha: The paramter alpha multiplied by -2 times beta.
    :type reduced_alpha: float

    :return: The current relative magnetisation.
    :rtype: float
    """
    number_of_traders = 2 * black.shape[0] * black.shape[1]
    global_market = np.sum(black + white)
    market_coupling = reduced_alpha * np.abs(global_market) / number_of_traders
    probabilities = precompute_probabilities(reduced_neighbour_coupling, market_coupling)
    update_strategies(True, black, white, probabilities)
    update_strategies(False, white, black, probabilities)

    return global_market / number_of_traders
