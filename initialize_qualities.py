"""
Tools for initializing the locations of a grid with quality values. Assumes
that a grid has already been populated with locations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import initialize_locations as il
from location import Location


def main():
    grid = il.initialize_grid(
        100, False, False, show_plot_log_scale_x=True, return_stats=False
    )
    grid = set_grid_quality(grid, False, False,
                            show_plot_log_scale_x=True, return_stats=False)

    grid = il.initialize_grid(
        100, 10, False, show_plot_log_scale_x=True, return_stats=False
    )
    grid = set_grid_quality(grid, 10, False,
                            show_plot_log_scale_x=True, return_stats=False)

    grid = il.initialize_grid(
        100, 10, 22, show_plot_log_scale_x=True, return_stats=False
    )
    grid = set_grid_quality(grid, 10, 22,
                            show_plot_log_scale_x=True, return_stats=False)

    grid = il.initialize_grid(
        100, 3, False, show_plot_log_scale_x=False, return_stats=False,
        capacity_fx=il.capacity_central_city
    )
    grid = set_grid_quality(
        grid, 3, False, show_plot_log_scale_x=False, return_stats=False
    )


def set_grid_quality(grid: np.ndarray,
                     smooth: int | bool = 5,
                     prune: int | bool = False,
                     quality_fx: callable = None,
                     show_grid: bool = True,
                     show_grid_log_scale: bool = True,
                     save_grid: bool | str = False,
                     show_plot: bool = True,
                     show_plot_log_scale_x: bool = True,
                     save_plot: bool | str = False,
                     return_stats: bool = True) \
        -> np.array | tuple[np.array, float, float, list[int]]:
    """
    Takes a 2 dimensional numpy array that has been populated with location
    instances and sets the quality of those location instances.
    Args:
        grid: The 2-dimensional numpy array of locations.
        smooth: Either False, indicating that no smoothing is to be
        performed, or an integer specifying the number of smoothing
        iterations to be performed.
        prune: Either False, indicating that no smoothing is to be performed,
        or an integer specifying the the capacity threshold below which nodes
        will be removed. Pruning is applied after smoothing.
        quality_fx: The function to use to determine what the initial
        quality of a location on the grid will be (prior to smoothing). Must
        take grid row index and column index as arguments, and return an
        float greater than zero. If None, then a simple power law
        distribution independent of row and column index will be applied,
        with an alpha of 5/2, to allow for a finite theoretical mean and
        variance.
        show_grid: Whether or not to show location quality as a heat map
        after grid initialization.
        show_grid_log_scale: Whether or not to transform quality with a
        logarithmic function before displaying it on the heatmap. If a power
        law or similar distribution is used to generate location quality,
        this may produce better results.
        save_grid: Whether or not to save the heatmap. If False, the heat map
        is not saved. If a string, then that string is used as the
        destination to save the file to.
        show_plot: Whether or not to display a rank-frequency distribution of
        location qualities after grid initialization.
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
        If return_stats is True, returns the initialized grid, mean quality,
        quality standard deviation, and a sorted list of qualities.

    """

    # Start Argument Validation #
    if not (isinstance(grid, np.ndarray) and len(grid.shape) == 2):
        raise ValueError('Grid must be a 2-dimensional numpy array.')
    if not grid.shape[0] == grid.shape[1]:
        raise ValueError('Grid must be square.')
    if not (isinstance(smooth, int) and smooth > 0) and \
            not (isinstance(smooth, bool) and smooth is False):
        raise ValueError("'smooth' must be either false or an integer greater "
                         "zero.")
    if not (isinstance(prune, int) and prune > 0) and \
            not (isinstance(prune, bool) and prune is False):
        raise ValueError("'prune' must be either false or an integer greater "
                         "zero.")
    if quality_fx is None:
        def quality_fx(_, __, ___, ____):
            return int((1 - np.random.rand()) ** (-1 / (5 / 2 - 1)) * 10)
    if not callable(quality_fx):
        raise ValueError("'quality_fx' must be callable.")
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

    size = grid.shape[0]
    for i in range(size):
        for j in range(size):
            if isinstance(grid[i][j], Location):
                grid[i][j].quality = \
                    quality_fx(i, j, size, grid[i][j].capacity)

    if smooth:
        grid = smooth_grid(grid, smooth)
    if prune:
        grid = prune_grid(grid, prune)

    def get_quality(element: Location | None):
        if element is None:
            return 0
        else:
            return element.quality

    get_quality_v = np.vectorize(get_quality)

    if show_grid or save_grid:
        fig_grid: plt.Figure = plt.figure(dpi=300)
        ax: plt.Axes = fig_grid.add_subplot()
        if show_grid_log_scale:
            ax.imshow(np.log(get_quality_v(grid)))
            ax.set_title('Quality Distribution, Log Scale')
        else:
            ax.imshow(get_quality_v(grid))
            ax.set_title('Quality Distribution, Linear Scale')
        if show_grid:
            fig_grid.show()
        if save_grid:
            fig_grid.savefig(save_grid)

    if show_plot or save_plot or return_stats:
        qualities = list(get_quality_v(grid.flatten()))
        qualities.sort(reverse=True)

    if show_plot or save_plot:
        fig_plot: plt.Figure = plt.figure(dpi=300)
        ax: plt.Axes = fig_plot.add_subplot()
        ax.scatter(list(range(len(qualities))), qualities, s=1)
        if show_plot_log_scale_x:
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Quality Rank-Size Distribution')
        if show_plot:
            fig_plot.show()
        if save_plot:
            fig_plot.savefig(save_plot)

    if not return_stats:
        return_value = grid
    else:
        mean = float(np.mean(qualities))
        std_dev = float(np.std(qualities))
        return_value = (grid, mean, std_dev, qualities)

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

    def get_quality(element: Location | None):
        if element is None:
            return 0
        else:
            return element.quality

    dim = grid.shape[0]
    for _ in range(iterations):
        for i in range(dim):
            for j in range(dim):
                if grid[i][j] is None:
                    continue
                values = list()
                values.append(get_quality(grid[i][j]))
                if i > 0:
                    values.append(get_quality(grid[i - 1][j]))
                if i < dim - 1:
                    values.append(get_quality(grid[i + 1][j]))
                else:
                    values.append(get_quality(grid[0][j]))
                if j > 0:
                    values.append(get_quality(grid[i][j - 1]))
                if j < dim - 1:
                    values.append(get_quality(grid[i][j + 1]))
                else:
                    values.append(get_quality(grid[i][0]))
                grid[i][j].quality = np.mean(values)
    for i in range(dim):
        for j in range(dim):
            if grid[i][j] is not None:
                grid[i][j].quality = np.floor(grid[i][j].quality)
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
            if grid[i][j].quality < threshold:
                grid[i][j] = None
    return grid


if __name__ == '__main__':
    main()
