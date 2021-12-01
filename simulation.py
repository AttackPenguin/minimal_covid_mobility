from __future__ import annotations

import os
import pickle
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from agent import Agent
from location import Location


class Simulation:

    def __init__(self,
                 loc_cost: int | float,
                 agent_move_cost: int | float,
                 agent_move_qual: int | float,
                 agent_move_covid: int | float,
                 agent_move_threshold: int | float,
                 agent_move_distance: int | float,
                 agent_score_cost: int | float,
                 agent_score_qual: int | float,
                 agent_score_covid: int | float,
                 grid_size: int,
                 loc_init_func: callable,
                 lif_kwargs: dict[str, Any],
                 agent_init_func: callable,
                 aif_kwargs: dict[str, Any],
                 qual_init_func: callable,
                 qif_kwargs: dict[str, Any]):

        # Parameter used by location class to calculate cost
        self.loc_cost = loc_cost

        # Parameters used by agent class to determine if agents will move
        self.agent_move_cost = agent_move_cost
        self.agent_move_covid = agent_move_covid  # 0 = no covid effect
        self.agent_move_qual = agent_move_qual

        # Parameter used in simulation class when it compares agent move
        # scores. Should be in the range [0, 1].
        self.agent_move_threshold = agent_move_threshold
        # Parameter used in simulation class when it decides how far to look
        # for new locations for agent to consider
        self.agent_move_distance = agent_move_distance

        # Parameters used by agent class to determine how much it likes a
        # location
        self.agent_score_cost = agent_score_cost
        self.agent_score_qual = agent_score_qual
        self.agent_score_covid = agent_score_covid

        # width / height of square grid on which simulation takes place
        self.grid_size = grid_size

        # Call the three initialization functions
        # First initialize location, passing keyword arguments as well
        self.grid = loc_init_func(grid_size, **lif_kwargs)
        # Then initialize quality, again with kwargs
        qual_init_func(self.grid, **qif_kwargs)
        # Then populate with agents, getting a dictionary of agents back,
        # with id values as keys
        _, self.agents = agent_init_func(self.grid, **aif_kwargs)

        # Initialize our time step to zero
        self.time_step = 0

        # Create a list of tuples, with 4 'columns' of data corresponding to:
        # agent id
        # row of location agent is at
        # column of location agent is at
        # current time step (0 at the moment)
        self.data = [
            (
                idx,
                self.agents[idx]['row'],
                self.agents[idx]['col'],
                self.time_step
            )
            for idx in self.agents
        ]

        # Create a list containing the unique salary values present in
        # self.agents to facilitate some other processes:
        self.salaries = list(self.agents[idx]['agent'].salary for idx in
                             self.agents)
        self.salaries.sort()

        # Create a dictionary to hold data on the number of moves per time
        # step by salary. Keys for the dictionary will be the unique salary
        # values of the agents. Values will be empty dictionaries. These
        # sub-dictionaries will be populated at each timestep with the
        # time step as the key of an entry and the number of agents moving
        # with that salary as the associated value.
        self.move_data = {
            salary: dict()
            for salary in self.salaries
        }

    def step_simulation(self,
                        agent_move_covid: int | float | None = None,
                        agent_score_covid: int | float | None = None):

        # Increment the timestep
        self.time_step += 1

        # See if covid values have changed, and if so, modify those variables
        # appropriately.
        if agent_score_covid is not None:
            self.agent_score_covid = agent_score_covid
        if agent_move_covid is not None:
            self.agent_move_covid = agent_move_covid

        # Update all of our location's agent counts. We need these values to
        # be updated before we can calculate whether agents move and how much
        # they like the locations they're considering moving to.
        pop_counts = [
            (self.agents[idx]['row'], self.agents[idx]['col'])
            for idx in self.agents
        ]
        pop_counts = Counter(pop_counts)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid[i][j].agents = pop_counts[(i, j)]

        # Get list of agents that are moving.
        moving_agents = self.get_moving_agents()

        # Get move locations and change agent location data
        for idx in moving_agents:
            move_loc = self.get_move_location(idx)
            self.agents[idx]['row'] = move_loc[0]
            self.agents[idx]['col'] = move_loc[1]

        # Update data by appending a new list of agent locations with the
        # timestep onto the existing one.
        self.update_data()

        # We also update our list of agents that have moved.
        self.update_move_data(moving_agents)

    def get_moving_agents(self):

        # Create a dictionary to hold move scores for agents. Will use agent
        # idx values for keys, and store scores as values.
        move_scores = dict()

        # Iterate through agents, calculating the move score for each based
        # on its current location, and adding idx and score to move_scores
        for idx in self.agents:
            move_scores[idx] = self.agents[idx]['agent'].get_move_score \
                    (
                    self.grid[self.agents[idx]['row']][self.agents[idx]['col']],
                    self.agent_move_cost,
                    self.agent_move_qual,
                    self.agent_move_covid
                )

        # Create a list to store idx values of agents that will be moving
        moving_agents = list()

        # Normalize move_scores by dividing each score by the maximum score,
        # then compare the normalize score to the move threshold parameter to
        # decide if an agent moves
        max_move_score = max(move_scores.values())
        for idx in move_scores:
            norm_move_score = move_scores[idx] / max_move_score
            if norm_move_score > self.agent_move_threshold:
                moving_agents.append(idx)

        # We return a list of idx values of agents that will move this time step
        return moving_agents

    def get_move_location(self,
                          idx: int):

        # Get the maximum distance to move. Multiply salary by factor and add
        # 1, so that all agents that decide to move have options. Convert to
        # int, which is equivalent to taking floor of value.
        move_distance = \
            int(1 + self.agents[idx]['agent'].salary * self.agent_move_distance)
        if move_distance > self.grid_size:
            raise Exception('Move distance generated is larger than width of '
                            'simulation grid...')

        # Get current position row and column
        row_current = self.agents[idx]['row']
        col_current = self.agents[idx]['col']

        # We're going to include all rows and columns within move_distance
        # steps of the current location.
        row_vals = \
            list(range(row_current - move_distance,
                       row_current + move_distance + 1))
        col_vals = \
            list(range(col_current - move_distance,
                       col_current + move_distance + 1))

        # For a small move range, we get a list of all locations within
        # reach, and then call numpy.random.choice.
        if move_distance <= 2:

            # Get a list of reachable grid coordinates, excluding current
            # location
            possible_locations = list()
            # We iterate through all reachable coordinates and add them to our
            # list of possibilities as tuples of rows and columns.
            for row in row_vals:
                for col in col_vals:
                    # Don't include our current location
                    if row == row_current and col == col_current:
                        continue
                    # Account for moves that take us off the right side or
                    # bottom of the grid, wrapping them around.
                    if row >= self.grid_size:
                        row = row - self.grid_size
                    elif row < 0:
                        row = self.grid_size + row
                    if col >= self.grid_size:
                        col = col - self.grid_size
                    elif col < 0:
                        col = self.grid_size + col
                    possible_locations.append((row, col))

            # Randomly pick 8 locations in list of possibilities. If we want,
            # we could readily add a parameter here, but iterating through these
            # locations is the highest cost portion of our simulation, and wrt
            # coding it, this is also a very convenient simplification.
            # Note that 8 locations is the Moore neighborhood for our lowest
            # socioeconomic class.
            loc_indices = np.random.choice(
                list(range(len(possible_locations))), 8, replace=False
            )
            # Put our chosen locations into a list.
            locations = [possible_locations[i] for i in loc_indices]

        # For a large move range, the grid of possibilities is very large,
        # so it's faster to randomly pick values in range and add them to a
        # list until we get eight unique tuples.
        else:

            locations = list()
            while len(locations) < 8:
                # Get random row and column values from range
                row = np.random.choice(row_vals)
                col = np.random.choice(col_vals)
                # If row and column are current location, get another location
                if (row, col) == (row_current, col_current):
                    continue
                # If values are out of range, adjust
                if row >= self.grid_size:
                    row = row - self.grid_size
                elif row < 0:
                    row = self.grid_size + row
                if col >= self.grid_size:
                    col = col - self.grid_size
                elif col < 0:
                    col = self.grid_size + col
                # If row and column are already in list, continue
                if (row, col) in locations:
                    continue
                # Otherwise, add location to locations.
                else:
                    locations.append((row, col))

        # Generate scores for each location in the list.
        scores = [
            self.agents[idx]['agent'].get_location_score(
                self.grid[row][col],
                self.agent_score_cost,
                self.agent_score_qual,
                self.agent_score_covid)
            for (row, col) in locations
        ]

        # Use index of max score to identify grid coordinates of highest
        # scoring location.
        move_location = locations[scores.index(max(scores))]

        # Return tuple of grid coordinates to move to
        return move_location

    def update_data(self):

        # Adds a new time step worth of data to the detailed agent data we
        # will return from our simulation.
        new_data = [
            (
                idx,
                self.agents[idx]['row'],
                self.agents[idx]['col'],
                self.time_step
            )
            for idx in self.agents
        ]
        self.data += new_data

    def update_move_data(self,
                         moved_agents: list[Agent]):

        # Adds a new time step worth of data to the agent movement data we
        # will return from our simulation.
        counter = Counter(
            [
                self.agents[idx]['agent'].salary
                for idx in moved_agents
            ]
        )
        for salary in self.salaries:
            if salary in counter:
                self.move_data[salary][self.time_step] = counter[salary]
            else:
                self.move_data[salary][self.time_step] = 0

    def store_data(self,
                   directory: str,
                   file_name: str):

        # Allows us to pass these values into our simulation directly,
        # rather than via __init__.
        destination = os.path.join(directory, file_name)

        # Convert self.data into a DataFrame. Experimentally, this results in
        # an insignificant increase in stored data size over creating a json
        # file or leaving it as a list of tuples, and it leaves the data
        # ready for analysis when it is unpickled.
        data = pd.DataFrame(
            self.data,
            columns=['idx', 'row', 'col', 'time_step']
        )

        # Create a vectorized numpy capacity accessor here to easily
        # generate a grid containing location capacities.
        def get_capacity(loc: Location):
            return loc.capacity

        v_get_capacity = np.vectorize(get_capacity)

        # Create a numpy grid containing the location capacities to return as
        # part of the pickled data.
        grid_capacity = v_get_capacity(self.grid)

        # Another vectorized accessor for location quality
        def get_quality(loc: Location):
            return loc.quality

        v_get_capacity = np.vectorize(get_quality)

        # Create a numpy grid of location quality to return.
        grid_quality = v_get_capacity(self.grid)

        # Create a list of agent salaries to return, ordered such that list
        # indices correspond to agent idx values.
        agent_salary = [
            self.agents[idx]['agent'].salary
            for idx in range(len(self.agents))
        ]

        # Pickle a file containing our data structures.
        # This is a VERY large file, due to the size of 'data',
        # which contains a row for every agent for every timestep.
        with open(destination, 'wb') as file:
            pickle.dump(
                (
                    data,
                    self.move_data,
                    grid_capacity,
                    grid_quality,
                    agent_salary
                ), file
            )
