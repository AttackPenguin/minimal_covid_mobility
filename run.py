import os
import pickle

import pandas as pd


from analysis_tools import plot_movement_rates
import simulation as sim
# We import our three initialization functions, one each for grid location,
# location quality, and for placing agents.
from initialize_locations import initialize_grid as init_locations
from initialize_qualities import set_grid_quality as init_quality
from initialize_agents import place_grid_agents as init_agents

dttm_start = pd.Timestamp.now()
print(f"Started at {dttm_start.strftime('%H:%M:%S')}\n")

# Create a simulation object, passing it all of our parameters.
simulation = sim.Simulation(

    # Location class parameters
    loc_cost=1,

    # Agent class parameters, some of which are also used directly by
    # simulation class.
    agent_move_cost=10_000,
    agent_move_qual=3_000,
    agent_move_covid=0,

    agent_move_threshold=0.009,  # Extremely sensitive in range [0.005, 0.01]
    agent_move_distance=1/10_000,

    agent_score_cost=1,
    agent_score_qual=10,
    agent_score_covid=0,

    # The width and height of our 2-dimensional simulation grid
    grid_size=100,

    # Our location initialization function, along with keyword parameters to
    # be passed to that function.
    loc_init_func=init_locations,
    lif_kwargs={
        'smooth': 10,
        'return_stats': False
    },

    # Our quality initialization function, along with keyword parameters to
    # be passed to that function.
    qual_init_func=init_quality,
    qif_kwargs={
        'smooth': 10,
        'return_stats': False
    },

    # Our agent initialization function, along with keyword parameters to be
    # passed to that function.
    agent_init_func=init_agents,
    aif_kwargs={}
)

# To advance the simulation one step, we call Simulation.step_simulation()
for i in range(1, 51):
    simulation.step_simulation()
    if i % 25 == 0:
        print(f"{i} iterations completed...")

# When we want to introduce covid to the simulation, we pass arguments that
# change the value of two paramaters that modify how agents decide to move
# and how they calculate the value of new locations.
simulation.step_simulation(agent_move_covid=1, agent_score_covid=2)
print("The pandemic is upon us...")

# We continue to advance the simulation.
for i in range(52, 101):
    simulation.step_simulation()
    if i % 25 == 0:
        print(f"{i} iterations completed...")

# When we are done running the simulation, we call Simulation.store_data() to
# export all simulation data to a file. This file is created using the pickle
# library, and consists of one EXTREMELY large 'tidy data' format pandas
# DataFrame that contains agent location information for every time step,
# and three smaller files that contain data on grid location quality,
# capacity, and agent salary.
directory = './Saved Data'
file_name = 'Test Run.pickle'
simulation.store_data(directory, file_name)

# When we want to analyze data, we unpickle the file to get our data
# structures back in a format that is ready for data analysis.
destination = os.path.join(directory, file_name)
with open(destination, 'rb') as file:
    data, move_data, grid_caps, grid_quals, agent_salaries = pickle.load(file)

plot_movement_rates(data, agent_salaries, move_data)

dttm_finished = pd.Timestamp.now()
print(f"Finished at {dttm_finished.strftime('%H:%M:%S')}")
run_time = dttm_finished-dttm_start
print(f"Run Time: {run_time.total_seconds()/60:.2f} Minutes")