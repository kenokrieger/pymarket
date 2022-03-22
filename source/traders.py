import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


@cuda.jit
def init_traders(is_black, rng_states, target, shape, init_up=0.5):
    """
    Flip 'init_up' percentage of spins on GPU.
    """
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    # Linearized thread index
    thread_id = (starty * stridex) + startx

    # Use strided loops over the array to assign a random value to each entry
    for row in range(startx, shape[0], stridex):
        for col in range(starty, shape[1], stridey):
            random = xoroshiro128p_uniform_float32(rng_states, thread_id)
            if random < init_up:
                target[row, col] = -1


def precompute_probabilities(probabilities, reduced_neighbor_coupling, market_coupling):
    """
    Precompute all possible values for the flip-probabilities.
    """
    for row in range(0, 2):
        spin = 1 if row else -1
        for col in range(5):
            neighbor_sum = -4 + 2 * col
            field = reduced_neighbor_coupling * neighbor_sum - market_coupling * spin
            probabilities[row * 5 + col] = 1 / (1 + np.exp(field))


@cuda.jit
def update_strategies(is_black, rng_states, source, checkerboard_agents, probabilities, shape):
    """
    Update all spins in one array according to the Heatbath dynamic.
    """
    startx, starty = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    # Linearized thread index
    thread_id = (starty * stridex) + startx

    # Use strided loops over the array to assign a random value to each entry
    for row in range(startx, shape[0], stridex):
        for col in range(starty, shape[1], stridey):
                row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
                col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
                spin = source[row, col]

                lower_neighbor_row = row + 1 if (row + 1 < shape[0]) else 0
                upper_neighbor_row = row - 1
                right_neighbor_col = col + 1 if (col + 1 < shape[1]) else 0
                left_neighbor_col = col - 1

                if is_black:
                    horizontal_neighbor_col = left_neighbor_col if row % 2 else right_neighbor_col
                else:
                    horizontal_neighbor_col = right_neighbor_col if row % 2 else left_neighbor_col

                neighbor_sum = (
                    checkerboard_agents[upper_neighbor_row, col]
                +   checkerboard_agents[lower_neighbor_row, col]
                +   checkerboard_agents[row, col]
                +   checkerboard_agents[row, horizontal_neighbor_col]
                )


                random = xoroshiro128p_uniform_float32(rng_states, thread_id)

                prob_row = 1 if spin + 1 else 0
                prob_col = (neighbor_sum + 4) // 2
                probability = probabilities[int(5 * prob_row + prob_col)]
                if random < probability:
                    source[row, col] = 1
                else:
                    source[row, col] = -1


@cuda.reduce
def sum_reduce(a, b):
    return a + b


def update(rng_states, black, white, reduced_neighbor_coupling, reduced_alpha, shape):
    """
    Update both arrays.
    """
    probabilities = np.empty((14, ), dtype=np.float32)
    threads_per_block = (16, 16)
    blocks = (32, 32)
    number_of_traders = 2 * shape[0] * shape[1]
    global_market = sum_reduce(black.ravel()) + sum_reduce(white.ravel())
    market_coupling = reduced_alpha * np.abs(global_market) / number_of_traders
    precompute_probabilities(probabilities, reduced_neighbor_coupling, market_coupling)

    update_strategies[blocks, threads_per_block](True, rng_states, black, white, probabilities, shape)
    cuda.synchronize()
    update_strategies[blocks, threads_per_block](False, rng_states, white, black, probabilities, shape)

    return global_market / number_of_traders
