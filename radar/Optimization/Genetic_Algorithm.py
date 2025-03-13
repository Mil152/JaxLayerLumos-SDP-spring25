import pygad
import jax.numpy as jnp
import numpy as np
from jaxlayerlumos import stackrt_eps_mu
from jaxlayerlumos import utils_materials
import matplotlib.pyplot as plt
import random

#---------INPUTS--------------------
nfreq = 500
layers = 5
thickness_target = 5*10**-3 #m
target_R_db = -30 #dB
freq_lowerbound = 0.2*10**9 #Hz
freq_upperbound = 2*10**9 #Hz
#---------------------------------------

# 0.1 GHz to 10 GHz, logarithmically spaced
frequencies = jnp.logspace(np.log10(freq_lowerbound), np.log10(freq_upperbound), nfreq)
# Same thing but for plotting
freqplot = jnp.logspace(np.log10(freq_lowerbound*1e-9), np.log10(freq_upperbound*1e-9), nfreq)

def fitness_func(ga_instance, solution, solution_idx):

#Run Sim
        mats = ["Air"]
        for i in range(layers):
            mats.append(solution[i])
        mats.append("PEC")
        materials = np.array(mats)

        thickness = [0]
        for i in range(5,10):
            thickness.append(solution[i])
        thickness.append(0)
        thicknesses = np.array(thickness)
        d_stack = jnp.array(thicknesses)
        total_thickness = jnp.sum(d_stack)


        #print(d_stack)
        #print(materials)

        eps_stack, mu_stack = utils_materials.get_eps_mu(materials, frequencies)
        # eps, mu, thick, freq, angle
        R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(eps_stack, mu_stack, d_stack, frequencies, 0.0)

        R_avg = (R_TE + R_TM) / 2
        R_db = 10 * jnp.log10(R_avg).squeeze()
        minRmax = max(R_db)


        # **Fitness Calculation with Normalization**
        reflection_error = abs(minRmax - target_R_db) / abs(target_R_db)  # Normalize
        thickness_error = abs(total_thickness - thickness_target) / thickness_target  # Normalize

        alpha = 0.8  # Weight for reflection
        beta = 0.2   # Weight for thickness constraint

        #fitness1 = 1/reflection_error
        #fitness2 = 1/thickness_error
        fitness2 = -1*minRmax
        fitness1 = -1*total_thickness
        return [np.array(fitness1).item(), np.array(fitness2).item()]

num_generations = 1000 # Number of generations.
num_parents_mating = 20 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 75 # Number of solutions in the population.
num_genes = 10

last_fitness = 0
def on_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


# Define the gene space with boundaries for material indices and thicknesses
gene_space = [{'low': 1, 'high': 16}] * layers + [{'low': 0.0002, 'high': 0.004}] * layers
mutation_adaptive = (0.4, 0.01)

ga_instance = pygad.GA(num_generations=num_generations,
                       parent_selection_type="nsga2",
                       K_tournament=3,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       fitness_func=fitness_func,
                       on_generation=on_generation,
                       keep_elitism=5,
                       mutation_probability=mutation_adaptive,  # Increase from default 0.1
                        crossover_type="scattered",  # More diverse offspring
                        mutation_type="adaptive",    # Ensure good exploration
                       gene_type = [int] * layers + [float] * layers)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

ga_instance.plot_fitness()
ga_instance.plot_pareto_front_curve(title ="GA Pareto Front", ylabel = "Reflection [dB]", xlabel = "Negative Thickness [m]")

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")



if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

# Saving the GA instance.
filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)

# Loading the saved GA instance.
#loaded_ga_instance = pygad.load(filename=filename)
#loaded_ga_instance.plot_fitness()



########--Ploting Results--#####################

#Run Sim
mats = ["Air"]
for i in range(layers):
    mats.append(solution[i])
mats.append("PEC")
materials = np.array(mats)
thickness = [0]
for i in range(5,10):
    thickness.append(solution[i])
thickness.append(0)
thicknesses = np.array(thickness)
d_stack = jnp.array(thicknesses)
total_thickness = jnp.sum(d_stack)
print(d_stack)
print(materials)

eps_stack, mu_stack = utils_materials.get_eps_mu(materials, frequencies)
# eps, mu, thick, freq, angle
R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(eps_stack, mu_stack, d_stack, frequencies, 0.0)

R_avg = (R_TE + R_TM) / 2
R_db = 10 * jnp.log10(R_avg).squeeze()
minRmax = max(R_db)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(freqplot, R_db, label='Best Sample')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Reflection Coefficient (dB)')
plt.title('Reflection vs frequency')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.legend()
plt.grid(True, which="both", ls="--")  # Grid for both major and minor ticks
plt.show()

print("Final Min dB", minRmax)
print("Final Thickness", total_thickness)