"""
Tools for populating an initialized grids Location objects with agents.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import initialize_locations as il
import initialize_qualities as iq

from location import Location
from agent import Agent


def main():
    grid = il.initialize_grid(
        100, 10, False, show_plot_log_scale_x=True, return_stats=False
    )
    grid = iq.set_grid_quality(grid, 10, False,
                               show_plot_log_scale_x=True, return_stats=False)
    grid, agents = place_grid_agents(grid)
    pass


def test_method_for_output_data():
    """
    Test method for working on analysis of simulation output. Will return an
    initialized grid and a list of agents. Agents are populated to 99% of
    capacity, and distributed in proportion to location capacity. With random
    movement of agents, this should provide a predictable outcome for setting up
    data visualization and analysis tools.
    """
    grid = il.initialize_grid(
        100, 10, False, show_plot_log_scale_x=True, return_stats=False
    )
    grid = iq.set_grid_quality(grid, 10, 22,
                               show_plot_log_scale_x=True, return_stats=False)
    return place_grid_agents(grid)


def place_grid_agents(grid: np.ndarray,
                      agent_fx: callable = None,
                      agent_fx_kwargs: dict[str, Any] | None = None,
                      return_stats: bool = True) \
        -> tuple[np.ndarray, dict[int, dict[str, int | Agent]]]:
    """
    Takes a 2 dimensional numpy array that has been populated with location
    instances and populates those instances with agents.
    Args:
        grid: The 2-dimensional numpy array of locations.
        agent_fx: A method that takes grid coordinates, and idx number for
        the first agent to be created, grid size, a location,
        and total grid capacity as arguments and populates the location with
        agents.
        agent_fx_kwargs: Keyword arguments to pass to agent_fx.
        return_stats: Whether or not to return summary statistics. At this time
        simply returns a list of agents.

    Returns:
        If return_stats is False, returns only the initialized grid.
        If return_stats is True, returns the initialized grid, and a list of
        agents.

    """
    # Deal with mutable default parameters #
    if agent_fx_kwargs is None:
        agent_fx_kwargs = {'percent_cap': 0.99}

    # Start Argument Validation #
    if not (isinstance(grid, np.ndarray) and len(grid.shape) == 2):
        raise ValueError('Grid must be a 2-dimensional numpy array.')
    if not grid.shape[0] == grid.shape[1]:
        raise ValueError('Grid must be square.')
    if agent_fx is None:
        agent_fx = default_agent_fx
    elif not callable(agent_fx):
        raise ValueError("'quality_fx' must be callable.")
    if not isinstance(return_stats, bool):
        raise ValueError("'return_stats' must be bool.")
    # End Argument Validation #

    size = grid.shape[0]
    total_cap = 0
    for i in range(size):
        for j in range(size):
            if isinstance(grid[i][j], Location):
                total_cap += grid[i][j].capacity

    agents = dict()
    idx = 0
    for i in range(size):
        for j in range(size):
            if isinstance(grid[i][j], Location):
                new_agents, idx = agent_fx(i, j, size, grid[i][j],
                                           total_cap, idx, **agent_fx_kwargs)
                for agent in new_agents:
                    agents[agent.idx] = {
                        'agent': agent,
                        'row': i,
                        'col': j
                    }

    return grid, agents


def default_agent_fx(i: int,
                     j: int,
                     grid_size: int,
                     location: Location,
                     total_cap: int,
                     idx_first: int,
                     percent_cap: float | int):
    salaries = [
        0, 15_000, 30_000, 60_000, 120_000, 240_000
    ]
    weights = [
        0.01, 0.19, 0.20, 0.20, 0.20, 0.20
    ]

    total_agents = int(location.capacity * percent_cap)
    agents = list()
    idx_current = idx_first
    for _ in range(total_agents):
        salary = np.random.choice(salaries, replace=True, p=weights)
        agents.append(Agent(
            idx_current, salary
        ))
        idx_current += 1

    return agents, idx_current


if __name__ == '__main__':
    main()
