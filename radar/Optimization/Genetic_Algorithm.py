import pygad
import jax.numpy as jnp
import numpy as np
from jaxlayerlumos import stackrt_eps_mu
from jaxlayerlumos import utils_materials
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import matplotlib.colors as mcolors
import random

#---------INPUTS--------------------
nfreq = 500
layers = 5
thickness_target = 5*10**-3 #m
target_R_db = -30 #dB
freq_lowerbound = 0.2*10**9 #Hz
freq_upperbound = 2*10**9 #Hz

#Hyper-Parameters
num_generations = 2000
num_parents_mating = 7
sol_per_pop = 40
num_genes = 10
last_fitness = 0
mutation_adaptive = (0.3, 0.05)

# Define the gene space
gene_space = [{'low': 1, 'high': 16}] * layers + [{'low': 0.0002, 'high': 0.004}] * layers

#---------------------------------------

# 0.1 GHz to 10 GHz, logarithmically spaced
frequencies = jnp.logspace(np.log10(freq_lowerbound), np.log10(freq_upperbound), nfreq)
# Same thing but for plotting
freqplot = jnp.logspace(np.log10(freq_lowerbound*1e-9), np.log10(freq_upperbound*1e-9), nfreq)

all_generations_fitness = []
avg_thickness_fitness_per_gen = []
avg_reflection_fitness_per_gen = []

def stacksolve(tlist,matsin,output):

    #make the thickness list
    stacklist=[]
    stacklist.append(0)
    for i in range(len(tlist)):
      stacklist.append(tlist[i])
    stacklist.append(0)
    d_stack = jnp.array(stacklist)

    #make the materials list
    mats=[]
    mats.append("Air")
    for i in range(len(matsin)):
        mats.append(str(matsin[i]))
    mats.append("PEC")

    #get eps, mu, and then solve the stack
    eps_stack, mu_stack = utils_materials.get_eps_mu(mats, frequencies)
    R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(eps_stack, mu_stack, d_stack, frequencies, 0.0) #eps, mu, thick, freq, angle

    R_avg = (R_TE + R_TM) / 2
    R_db = 10 * jnp.log10(R_avg).squeeze()
    #output 1 is for the objective funciton and output 2 is for the whole frequency range
    if output==1:
      return max(R_db)
    if output==2:
      return R_db


def fitness_func(ga_instance, solution, solution_idx):

    tlist = solution[5:10]
    total_thickness = np.sum(tlist)

    minRmax = stacksolve(tlist,solution[0:5],1)

    fitness1 = -1 * total_thickness
    fitness2 = -1 * minRmax

    return [np.array(fitness1).item(), np.array(fitness2).item()]


def on_generation(ga_instance):
    global last_fitness, all_generations_fitness,mean_generation_fitness
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    all_generations_fitness.extend(ga_instance.last_generation_fitness)

    # Get the fitness values of the current generation
    current_fitness = np.array(ga_instance.last_generation_fitness)  # Shape: (pop_size, 2)

    # Compute mean fitness for both objectives
    avg_thickness = -1*np.mean(current_fitness[:, 0])  # Avg fitness for thickness
    avg_reflection = -1*np.mean(current_fitness[:, 1])  # Avg fitness for reflection

    # Store them
    avg_thickness_fitness_per_gen.append(avg_thickness)
    avg_reflection_fitness_per_gen.append(avg_reflection)

    print(f"Generation {ga_instance.generations_completed}:")
    print(f"  Avg Thickness Fitness = {avg_thickness:.4f}")
    print(f"  Avg Reflection Fitness = {avg_reflection:.4f}")



#------------ Running GA -------------------------------#

ga_instance = pygad.GA(num_generations=num_generations,
                       parent_selection_type="tournament_nsga2",
                       K_tournament=7,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       fitness_func=fitness_func,
                       on_generation=on_generation,
                       keep_elitism=5,
                       mutation_probability=mutation_adaptive,
                       crossover_type="scattered",
                       mutation_type="adaptive",
                       gene_type=[int] * layers + [float] * layers)

# Run GA
ga_instance.run()

ga_instance.plot_fitness()

#-------------- Extracting Data -----------------------------------#

# Extract Pareto front data
pop_fitness = np.array(all_generations_fitness)


thickness_fitness = -pop_fitness[:, 0]  # Convert back to positive dB
reflection_fitness = -pop_fitness[:, 1] # Convert back to meters

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

pareto_mask = is_pareto_efficient(pop_fitness)
pareto_reflection = reflection_fitness[pareto_mask]
pareto_thickness = thickness_fitness[pareto_mask]


# Generate a color gradient based on the generation index
num_points = len(avg_thickness_fitness_per_gen)
generation_indices = np.linspace(0, 1, num_points)  # Normalize between 0 and 1

# Choose a colormap ('viridis' works well, or try 'plasma')
cmap = cm.get_cmap('viridis')
colors = cmap(generation_indices)

# Plot Pareto Front & Additional Curves
plt.figure(figsize=(8, 6))

plt.scatter(avg_thickness_fitness_per_gen, avg_reflection_fitness_per_gen, c=generation_indices, cmap='viridis', alpha=0.7, label="Mean Points")
plt.scatter(pareto_thickness, pareto_reflection, color='r', label="Pareto Front")
#EEE paper LF thicknesses and reflections
paperLFx=[5.512*1e-3,3.588*1e-3,2.934*1e-3,2.478*1e-3]
paperLFy=[-33,-21,-18,-14]

plt.plot(paperLFx, paperLFy, 'r--', label="Literature LF")
plt.scatter(paperLFx, paperLFy, color='b', )

plt.xlabel("Total Thickness (m)")
plt.ylabel("Max Reflection (dB)")
plt.title("Pareto Front & Additional Curves")
plt.legend()
plt.grid(True)
plt.show()

# Save GA instance
ga_instance.save(filename='genetic')

# Extract the Pareto front solutions from the population

pareto_solutions = ga_instance.population[pareto_mask]

# Split into materials and thicknesses
pareto_materials = pareto_solutions[:, 0:5].astype(int)  # First 5 genes: materials
pareto_thicknesses = pareto_solutions[:, 5:10]  # Last 5 genes: thicknesses

# Save to a CSV file
pareto_df = pd.DataFrame({
    "Material 1": pareto_materials[:, 0],
    "Material 2": pareto_materials[:, 1],
    "Material 3": pareto_materials[:, 2],
    "Material 4": pareto_materials[:, 3],
    "Material 5": pareto_materials[:, 4],
    "Thickness 1 (m)": pareto_thicknesses[:, 0],
    "Thickness 2 (m)": pareto_thicknesses[:, 1],
    "Thickness 3 (m)": pareto_thicknesses[:, 2],
    "Thickness 4 (m)": pareto_thicknesses[:, 3],
    "Thickness 5 (m)": pareto_thicknesses[:, 4],
    "Total Thickness (m)": np.sum(pareto_thicknesses, axis=1),
    "Max Reflection (dB)": pareto_reflection
})

# Save as CSV file
pareto_df.to_csv("pareto_front_data.csv", index=False)

print("Pareto front materials and layer thicknesses saved to 'pareto_front_data.csv'")
