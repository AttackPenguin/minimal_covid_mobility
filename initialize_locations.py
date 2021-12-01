"""
Tools for initializing a grid with Locations and setting those location's
capacities.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from location import Location


def main():
    # With no smoothing, we get a perfect power law distribution for our
    # capacities, but there is no overall structure spatially:
    grid = initialize_grid(
        100, False, False, show_plot_log_scale_x=True, return_stats=False
    )

    # When we add smoothing, our distribution becomes somewhat skewed,
    # but acquires structure spatially:
    grid = initialize_grid(
        100, 10, False, show_plot_log_scale_x=True, return_stats=False
    )

    # Pruning can be added to remove small capacity locations, which restores
    # much of the power law structure, and simulates uninhabited areas. This
    # may or may not be compatible with our method of identifying locations
    # to move to:
    grid = initialize_grid(
        100, 10, 22, show_plot_log_scale_x=True, return_stats=False
    )

    # Other capacity distribution functions can be passed in, for example,
    # to simulate a large central city:
    grid = initialize_grid(
        100, False, False, show_plot_log_scale_x=False, return_stats=False,
        capacity_fx=capacity_central_city
    )

    # The same function, with smoothing:
    grid = initialize_grid(
        100, 3, False, show_plot_log_scale_x=False, return_stats=False,
        capacity_fx=capacity_central_city
    )

    # The same function, with smoothing and pruning:
    grid = initialize_grid(
        100, 3, 150, show_plot_log_scale_x=False, return_stats=False,
        capacity_fx=capacity_central_city
    )


def initialize_grid(size: int = 100,
                    smooth: int | bool = 5,
                    prune: int | bool = False,
                    location=Location,
                    dist_fx: callable = None,
                    capacity_fx: callable = None,
                    show_grid: bool = True,
                    show_grid_log_scale: bool = True,
                    save_grid: bool | str = False,
                    show_plot: bool = True,
                    show_plot_log_scale_x: bool = True,
                    save_plot: bool | str = False,
                    return_stats: bool = True) \
        -> np.array | tuple[np.array, float, float, list[int]]:
    """
    Generates a square numpy array and populates it with location instances
    with capacity values set.
    Args:
        size: The number of rows and columns in the grid. If smoothing is not
        performed, complexity of grid initialization is O(size^2). However,
        if smoothing is performed, complexity increases much more rapidly
        with size, and large grids may take a very long time to initialize.
        smooth: Either False, indicating that no smoothing is to be
        performed, or an integer specifying the number of smoothing
        iterations to be performed.
        prune: Either False, indicating that no smoothing is to be performed,
        or an integer specifying the the capacity threshold below which nodes
        will be removed. Pruning is applied after smoothing.
        location: The class to populate the nodes with. Must be Location or a
        class inheriting from location, and must implement get_capacity and
        set_capacity methods.
        dist_fx: The function to use to determine whether or not a grid
        location will be populated with a location instance. Must take grid
        row index and column index as arguments, and return True or False.
        If None, then all grid coordinates are populated with location
        instances.
        capacity_fx: The function to use to determine what the initial
        capacity of a location on the grid will be (prior to smoothing). Must
        take grid row index and column index as arguments, and return an
        integer greater than zero. If None, then a simple power law
        distribution independent of row and column index will be applied,
        with an alpha of 5/2, to allow for a finite theoretical mean and
        variance.
        show_grid: Whether or not to show location capacity as a heat map
        after grid initialization.
        show_grid_log_scale: Whether or not to transform capacity with a
        logarithmic function before displaying it on the heatmap. If a power
        law or similar distribution is used to generate location capacity,
        this may produce better results.
        save_grid: Whether or not to save the heatmap. If False, the heat map
        is not saved. If a string, then that string is used as the
        destination to save the file to.
        show_plot: Whether or not to display a rank-frequency distribution of
        location capacities after grid initialization.
        show_plot_log_scale_x: Whether or not to apply a log scale to the
        x-axis of the rank-frequency plot.
        save_plot: Whether or not to save the rank-frequency plot. If False,
        the figure is not saved. If a string, then that string is used as the
        destination to save the file to.
        return_stats: Whether or not to return summary statistics (mean and
        standard deviation) of location magnitudes along with a sorted list
        of magnitudes.

    Returns:
        If return_stats is False, returns only the initialized grid.
        If return_stats is True, returns the initialized grid, mean capacity,
        capacity standard deviation, and a sorted list of capacities.

    """

    # Start Argument Validation #
    if not (isinstance(size, int) and size > 1):
        raise ValueError('The size of the grid must be an integer greater '
                         'than Zero.')
    if not (isinstance(smooth, int) and smooth > 0) and \
            not (isinstance(smooth, bool) and smooth is False):
        raise ValueError("'smooth' must be either false or an integer greater "
                         "zero.")
    if not (isinstance(prune, int) and prune > 0) and \
            not (isinstance(prune, bool) and prune is False):
        raise ValueError("'prune' must be either false or an integer greater "
                         "zero.")
    if not issubclass(location, Location):
        raise ValueError("'location' is not an instance of a Location class.")
    if dist_fx is None:
        def dist_fx(_, __, ___):
            return True
    if not callable(dist_fx):
        raise ValueError("'dist_fx' must be callable.")
    if capacity_fx is None:
        def capacity_fx(_, __, ___):
            return int((1 - np.random.rand()) ** (-1 / (5 / 2 - 1)) * 10)
    if not callable(capacity_fx):
        raise ValueError("'capacity_fx' must be callable.")
    if not isinstance(show_grid, bool):
        raise ValueError("'show_grid' must be of type bool.")
    if not isinstance(show_grid_log_scale, bool):
        raise ValueError("'show_grid_log_scale' must be of type bool.")
    if isinstance(save_grid, bool) and save_grid is not False:
        if not isinstance(save_grid, str):
            raise ValueError("'save_grid' must be either False, or a string "
                             "specifying a save file destination and name.")
    if not isinstance(show_plot, bool):
        raise ValueError("'show_plot' must be of type bool.")
    if not isinstance(show_plot_log_scale_x, bool):
        raise ValueError("'show_plot_log_scale' must be of type bool.")
    if isinstance(save_plot, bool) and save_plot is not False:
        if not isinstance(save_plot, str):
            raise ValueError("'save_plot' must be either False, or a string "
                             "specifying a save file destination and name.")
    if not isinstance(return_stats, bool):
        raise ValueError("'return_stats' must be bool.")
    # End Argument Validation #

    grid_init: list[list] = \
        [[None for _ in range(size)] for _ in range(size)]
    idx = 0
    for i in range(size):
        for j in range(size):
            if dist_fx(i, j, size):
                grid_init[i][j] = location(
                    capacity_fx(i, j, size)
                )
                idx += 1
    grid: np.array = np.array(grid_init)

    if smooth:
        grid = smooth_grid(grid, smooth)
    if prune:
        grid = prune_grid(grid, prune)

    def get_capacity(element: Location | None):
        if element is None:
            return 0
        else:
            return element.capacity

    get_capacity_v = np.vectorize(get_capacity)

    if show_grid or save_grid:
        fig_grid: plt.Figure = plt.figure(dpi=300)
        ax: plt.Axes = fig_grid.add_subplot()
        if show_grid_log_scale:
            ax.imshow(np.log(get_capacity_v(grid)))
            ax.set_title('Capacity Distribution, Log Scale')
        else:
            ax.imshow(get_capacity_v(grid))
            ax.set_title('Capacity Distribution, Linear Scale')
        if show_grid:
            fig_grid.show()
        if save_grid:
            fig_grid.savefig(save_grid)

    if show_plot or save_plot or return_stats:
        capacities = list(get_capacity_v(grid.flatten()))
        capacities.sort(reverse=True)

    if show_plot or save_plot:
        fig_plot: plt.Figure = plt.figure(dpi=300)
        ax: plt.Axes = fig_plot.add_subplot()
        ax.scatter(list(range(len(capacities))), capacities, s=1)
        if show_plot_log_scale_x:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Capacity Rank-Size Distribution')
        if show_plot:
            fig_plot.show()
        if save_plot:
            fig_plot.savefig(save_plot)

    if not return_stats:
        return_value = grid
    else:
        mean = float(np.mean(capacities))
        std_dev = float(np.std(capacities))
        return_value = (grid, mean, std_dev, capacities)

    return return_value


def smooth_grid(grid: np.array,
                iterations: int) -> np.array:
    """
    Iterates through a grid of locations, calculating, at each location,
    the average of the location's capacity and it's neighbor's capacities,
    then setting the location's capacity to that value. Asynchronous.
    Args:
        grid: A 2-dimensional square numpy array, with locations that are
        either None or a Location class instance. Instances must implement
        get_capacity and set_capacity methods.
        iterations: Number of rounds of smoothing to perform.

    Returns:
        Returns the smoothed grid. However, note that as a copy is not made,
        this method will also modify the grid in place. If it is not desired
        to alter the original grid, be sure to pass a deep copy.

    """

    def get_capacity(element: Location | None):
        if element is None:
            return 0
        else:
            return element.capacity

    dim = grid.shape[0]
    for _ in range(iterations):
        for i in range(dim):
            for j in range(dim):
                if grid[i][j] is None:
                    continue
                values = list()
                values.append(get_capacity(grid[i][j]))
                if i > 0:
                    values.append(get_capacity(grid[i - 1][j]))
                if i < dim - 1:
                    values.append(get_capacity(grid[i + 1][j]))
                else:
                    values.append(get_capacity(grid[0][j]))
                if j > 0:
                    values.append(get_capacity(grid[i][j - 1]))
                if j < dim - 1:
                    values.append(get_capacity(grid[i][j + 1]))
                else:
                    values.append(get_capacity(grid[i][0]))
                grid[i][j].capacity = np.mean(values)
    for i in range(dim):
        for j in range(dim):
            grid[i][j].capacity = np.floor(grid[i][j].capacity)
    return grid


def prune_grid(grid: np.array,
               threshold: int) -> np.array:
    """
    For all Location instances populated on the passed grid, if their
    capacity is less than threshold, removes them, setting the grid element
    to None.
    Args:
        grid: A 2-dimensional square numpy array, with locations that are
        either None or a Location class instance. Instances must implement
        get_capacity method.
        threshold: Capacity under which to prune Location instances from grid.

    Returns:
        Returns the smoothed grid. However, note that as a copy is not made,
        this method will also modify the grid in place. If it is not desired
        to alter the original grid, be sure to pass a deep copy.

    """
    dim = grid.shape[0]
    for i in range(dim):
        for j in range(dim):
            if grid[i][j] is None:
                continue
            if grid[i][j].capacity < threshold:
                grid[i][j] = None
    return grid


def capacity_central_city(row: int,
                          column: int,
                          size: int) -> int:
    """
    A capacity_fx to be passed as an argument to initialize_grid. Creates an
    exponential radial distribution of capacities with the maximum at the
    center of the grid.
    Args:
        row: The row index of the grid position being populated.
        column: The column index of the grid position being populated.
        size: The width of the square grid.

    Returns:
        An integer capacity

    """
    if row >= size:
        raise IndexError("Row index is greater than number of rows in grid.")
    if row >= size:
        raise IndexError("Column index is greater than number of columns in "
                         "grid.")
    if size < 1:
        raise ValueError("'size' must be an integer greater than zero.")

    distance = np.sqrt(
        (np.abs(row - (0.5 * size))) ** 2 + (np.abs(column - (0.5 * size))) ** 2
    ) / np.sqrt(2)

    capacity = int((((size / (np.abs(distance) + (1 / 10) * size)) *
                     (np.random.rand() * .2 + .5)) * 10) ** 2)

    return capacity


if __name__ == '__main__':
    main()
