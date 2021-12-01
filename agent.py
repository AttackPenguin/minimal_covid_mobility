from __future__ import annotations

from location import Location


class Agent:

    def __init__(self,
                 idx: int,
                 salary: int | float):

        # A unique value for each agent, used to track agent movement
        # throughout the simulation.
        self.idx = idx
        # The agent's salary, which does not change over the course of the
        # simulation. Used to calculate movement and location scores in the
        # agent class, and in the simulation to calculate move distance.
        self.salary = salary

    def get_move_score(self,
                       loc: Location,
                       cost_var: int | float,
                       qual_var: int | float,
                       covid_var: int | float):
        # Score is the sum of three different factors, each multiplied by a
        # simulation parameter so that it can be tuned.
        move_score = (
            # High cost at current location increases probability of move
            # High salary makes cost less relevant.
            # Addition of 10,000 to denominator prevents extreme preference
            # for 0 income agents to move.
            cost_var * (loc.get_cost() / (self.salary+1000)) +
            # With no covid, cost_var is set to zero - no effect
            # With covid, high capacity increases likelihood of move as a
            # surrogate for density.
            covid_var * loc.capacity +
            # The higher the quality, the lower the probability of moving
            qual_var * (1 / loc.quality)
        )

        return move_score

    def get_location_score(self,
                           loc: Location,
                           cost_var: int | float,
                           qual_var: int | float,
                           covid_var: int | float) -> int | float:

        score = (
            # High salary increases score - cost is less relevant
            # High cost decreases score - cost is more relevant
            # Indirectly results in people with same salary drawn to same
            # locations.
            cost_var * (self.salary / loc.get_cost()) +
            # With no covid, variable is set to zero - no effect
            # With covid, lower capacity gets scored higher - capacity is a
            # surrogate for density.
            covid_var * (1 / loc.capacity) +
            # High quality always increase score
            qual_var * loc.quality
        )

        return score
