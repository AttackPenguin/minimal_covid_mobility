from __future__ import annotations


class Location:

    def __init__(self,
                 capacity: int,
                 quality: int | float | None = None,
                 agents: int = 0):
        # The amount of 'housing'. Beyond this amount, cost increases
        # rapidly. Also a surrogate for density of area. Low capacity
        # increases the agent location score when the pandemic variable is
        # not zero.
        self.capacity = capacity
        # A measure of the desirability of the area as a place to live.
        self.quality = quality
        # The number of agents currently at a location. Used by location to
        # calculate cost, and directly and indirectly by agents to generate
        # movement and desirability scores.
        self.agents = agents

    def get_cost(self,
                 cost_var: int | float = 1):
        # If area is not overpopulated, cost = quality
        if self.agents <= self.capacity:
            cost = self.quality
        # If area is overpopulated, cost = quality plus a rapidly increasing
        # factor proportional to overpopulation, quality, and cost_var
        else:
            cost = (
                self.quality +
                (self.agents / self.capacity) * self.quality * cost_var
            )
        return cost

