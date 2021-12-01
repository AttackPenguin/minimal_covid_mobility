from __future__ import annotations

import os
import pickle
from copy import copy, deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main():
    dttm_start = pd.Timestamp.now()
    print(f"Started at {dttm_start.strftime('%H:%M:%S')}\n")

    directory = './Saved Data'
    file_name = 'Test Run.pickle'
    destination = os.path.join(directory, file_name)
    with open(destination, 'rb') as file:
        data, move_data, grid_caps, grid_quals, agent_salaries = \
            pickle.load(file)

    plot_movement_rates(data, agent_salaries, move_data)

    dttm_finished = pd.Timestamp.now()
    print(f"Finished at {dttm_finished.strftime('%H:%M:%S')}")
    run_time = dttm_finished - dttm_start
    print(f"Run Time: {run_time.total_seconds() / 60:.2f} Minutes")


def add_salary_column_to_data(data: pd.DataFrame,
                              agent_salaries: list[int | float]) -> \
        pd.DataFrame:
    # Create a mapping from idx values to salary values
    mapping = {i: agent_salaries[i] for i in range(len(agent_salaries))}
    # Apply that map to the dataframe to create a new column with salary data
    data['salary'] = data['idx'].map(mapping)
    # We'll return data for flexibility, but this code modifies the passed
    # dataframe in place, so it's not generally necessary to use the return
    # value.
    return data


def get_num_agents_by_salary(data: pd.DataFrame,
                             agent_salaries: list[int | float]):
    if 'salary' not in data.columns:
        add_salary_column_to_data(data, agent_salaries)
    return data['salary'].value_counts().to_dict()


def plot_movement_rates(data: pd.DataFrame,
                        agent_salaries: list[int | float],
                        move_data: dict[int, dict[int, int]]):
    num_agents_by_salary = get_num_agents_by_salary(data, agent_salaries)

    norm_move_data = deepcopy(move_data)
    for salary in norm_move_data:
        for step in norm_move_data[salary]:
            norm_move_data[salary][step] /= num_agents_by_salary[salary]

    time_steps = max(data['time_step'])

    fig: plt.Figure = plt.figure(figsize=(6, 4), dpi=300)
    ax: plt.Axes = fig.add_subplot()
    for salary in norm_move_data:
        ax.plot(list(range(1, time_steps+1)),
                list(norm_move_data[salary].values()),
                label=salary)
    # ax.set_ylim(0, 1)
    ax.legend()
    fig.show()


if __name__ == '__main__':
    main()
