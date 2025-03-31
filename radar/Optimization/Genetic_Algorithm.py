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
import jax



#---------INPUTS--------------------
nfreq = 500
layers = 5
freq_lowerbound = 0.2*10**9 #Hz
freq_upperbound = 2*10**9 #Hz

#Hyper-Parameters
num_generations = 4000
num_parents_mating = 6
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
all_solutions = []
avg_thickness_fitness_per_gen = []
avg_reflection_fitness_per_gen = []
pareto_reflection = []
pareto_thickness =[]
pareto_thicknesses = []
pareto_materials =[]


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
    try:
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


    except Exception as e:
        #Catch any exception and return a default value
        print(f"Error in stacksolve: {e}")
        return float('inf')

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
    all_solutions.extend(ga_instance.population)

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
                       K_tournament=12,
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



thickness_fitness = -pop_fitness[:, 0]  # Convert back to negative dB
reflection_fitness = -pop_fitness[:, 1] # Convert back to meters

def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient
def ref4grad(tlist,matsin):
    return stacksolve(tlist,matsin,1)
pareto_mask = is_pareto_efficient(pop_fitness)
pareto_reflection = reflection_fitness[pareto_mask]
pareto_thickness = thickness_fitness[pareto_mask]
# Extract the Pareto front solutions from the population

pareto_solutions = np.array(all_solutions)[pareto_mask]
# Split into materials and thicknesses

pareto_materials = [[] for _ in range(len(pareto_solutions))]
pareto_thicknesses = [[] for _ in range(len(pareto_solutions))]

for i in range(len(pareto_solutions)):
    for j in range(0,5):
        pareto_materials[i].append((pareto_solutions[i,j]))
for i in range(len(pareto_solutions)):
    for j in range(5, 10):
        pareto_thicknesses[i].append((pareto_solutions[i,j]))

pareto_materials = np.array(pareto_materials, dtype=int)
pareto_thicknesses = np.array(pareto_thicknesses, dtype=float)

        # First 5 genes: materials
print(pareto_thicknesses)
print(pareto_materials)
c_pareto_ref=[] #copy pareto points to mess with
c_pareto_laythick=[]
c_pareto_totthick=[]


for i in range(len(pareto_reflection)):
    c_pareto_ref.append(pareto_reflection[i])
    c_pareto_laythick.append(pareto_thicknesses[i])
    c_pareto_totthick.append(pareto_thickness[i])

for i in range(len(pareto_reflection)):
    print(round(i/len(pareto_reflection)*100,4),"percent done with gradient descent")
    cont1=1
    for k in range(15):
        #R_db=stacksolve(parlayerthick[i],paretomats[i],2)

        gradients=jax.grad(ref4grad, argnums=0)
        gradtlist=gradients(pareto_thicknesses[i],pareto_materials[i])

        #print(f"Gradients for individual layers: {gradtlist}")
        c_pareto_ref[i]=ref4grad(pareto_thicknesses[i],pareto_materials[i])
        for j in range(len(gradtlist)): #all 5 layers
            grad=gradtlist[j]#gradient for a current layer
            c_pareto_laythick[i][j]-=grad*1e-8 #descend a little bit
            c_pareto_totthick[i]=sum(pareto_thicknesses[i])
grad_pop = np.column_stack([-1 * np.array(c_pareto_totthick, dtype=np.float64), -1 * np.array(c_pareto_ref, dtype=np.float64)])

grad_pop_fitness = np.append(pop_fitness, grad_pop, axis=0)  # Ensure correct axis
grad_reflection_fitness = -1*grad_pop_fitness[:, 1]
grad_thickness_fitness = -1*grad_pop_fitness[:, 0]
grad_pareto_mask = is_pareto_efficient(grad_pop_fitness)
grad_pareto_pop = grad_pop_fitness[grad_pareto_mask]

grad_pareto_reflection = grad_reflection_fitness[grad_pareto_mask]
grad_pareto_thickness = grad_thickness_fitness[grad_pareto_mask]

common_ref = np.intersect1d(np.array(c_pareto_ref, dtype=np.float64), grad_pareto_reflection)
common_thick = np.intersect1d(np.array(c_pareto_totthick, dtype=np.float64), grad_pareto_thickness)

#print(c_pareto_totthick)
#print(c_pareto_ref)
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
plt.scatter(common_thick, common_ref, color='m', label="Gradient Pareto Front")
#EEE paper LF thicknesses and reflections
paperLFx=[5.512*1e-3,3.588*1e-3,2.934*1e-3,2.478*1e-3]
paperLFy=[-33,-21,-18,-14]

plt.plot(paperLFx, paperLFy, 'r--', label="Literature LF")
plt.scatter(paperLFx, paperLFy, color='b', )

plt.xlabel("Total Thickness (m)")
plt.ylabel("Max Reflection (dB)")
plt.title("Pareto Front")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))

plt.scatter(avg_thickness_fitness_per_gen, avg_reflection_fitness_per_gen, c=generation_indices, cmap='viridis', alpha=0.7, label="Mean Points")
plt.scatter(pareto_thickness, pareto_reflection, color='r', label="Pareto Front")
plt.scatter(c_pareto_totthick, c_pareto_ref, color='m', label="Gradient Points")
#EEE paper LF thicknesses and reflections
paperLFx=[5.512*1e-3,3.588*1e-3,2.934*1e-3,2.478*1e-3]
paperLFy=[-33,-21,-18,-14]

plt.plot(paperLFx, paperLFy, 'r--', label="Literature LF")
plt.scatter(paperLFx, paperLFy, color='b', )

plt.xlabel("Total Thickness (m)")
plt.ylabel("Max Reflection (dB)")
plt.title("Pareto Front")
plt.legend()
plt.grid(True)
plt.show()

# Save GA instance
ga_instance.save(filename='genetic')





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
