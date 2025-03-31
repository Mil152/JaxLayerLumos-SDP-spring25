# materials
import jaxlayerlumos
import jax.numpy as jnp
import numpy as np
from jaxlayerlumos import stackrt_eps_mu
from jaxlayerlumos import utils_materials
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker

# load excel file
xls = pd.ExcelFile("Real_Materials.xlsx")

# 0.1 GHz to 10 GHz, logarithmically spaced
frequencies = jnp.logspace(np.log10(.1 * 10 ** 9), np.log10(10 * 10 ** 9), 200)
# same thing but for plotting
freqplot = jnp.logspace(np.log10(0.1), np.log10(10), 200)

materials = np.array(["Air", "4", "6", "8", "1", "2", "PEC"])
d_stack = jnp.array([0, 3.25, .26, .1, .5, .2, 0]) * 10 ** -3


########

def get_eps_mu_numpy(materials, frequencies, xls):
    """
    Retrieves permittivity and permeability from the provided Excel file using NumPy.

    Parameters:
    - materials: List of material indices as strings (e.g., ["1", "2", "PEC"])
    - frequencies: Array of frequencies in Hz

    Returns:
    - eps_stack: Complex permittivity array for each material at given frequencies
    - mu_stack: Complex permeability array for each material at given frequencies
    """
    eps_stack = []
    mu_stack = []
    for i in range(len(frequencies)):
        epsadd = []
        muadd = []
        for k in range(len(materials)):
            if materials[k] == "Air":
                epsadd.append(1 - 0.j)
                muadd.append(1 + 0j)
            if materials[k] == "PEC":
                epsadd.append(np.inf + 0.j)
                muadd.append(1 + 0j)
            if materials[k] != "Air" and materials[k] != "PEC":
                material_df = xls.parse(materials[k])

                freq_material = material_df["Frequency_GHz"].values * 1e9
                real_eps = material_df["Real_Epsilon"].values
                imag_eps = material_df["Imag_Epsilon"].values
                real_mu = material_df["Real_Mu"].values
                imag_mu = material_df["Imag_Mu"].values

                # able to interpolate these things as a function of freqency
                interp_real_eps = interp1d(freq_material, real_eps, kind="linear", fill_value="extrapolate")
                interp_imag_eps = interp1d(freq_material, imag_eps, kind="linear", fill_value="extrapolate")
                interp_real_mu = interp1d(freq_material, real_mu, kind="linear", fill_value="extrapolate")
                interp_imag_mu = interp1d(freq_material, imag_mu, kind="linear", fill_value="extrapolate")
                eps_complex = interp_real_eps(frequencies[i]) + 1j * interp_imag_eps(frequencies)
                mu_complex = interp_real_mu(frequencies[i]) + 1j * interp_imag_mu(frequencies)

                epsadd.append(eps_complex[0])
                muadd.append(mu_complex[0])

        eps_stack.append(epsadd)
        mu_stack.append(muadd)
    return np.array(eps_stack), np.array(mu_stack)


eps_stack, mu_stack = utils_materials.get_eps_mu(materials, frequencies)
# print(eps_stack)
print("#" * 20)
eps_stack_test, mu_stack_test = get_eps_mu_numpy(materials, frequencies, xls)
print("#" * 20)
# print(eps_stack_test)
# eps, mu, thick, freq, angle
R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(eps_stack, mu_stack, d_stack, frequencies, 0.0)

R_avg = (R_TE + R_TM) / 2

R_db_HF2 = 10 * jnp.log10(R_avg).squeeze()

# Plot results
import matplotlib.ticker as ticker

plt.figure(figsize=(8, 5))
plt.plot(freqplot, R_db_HF2, color=(170 / 255, 0 / 255, 1), linewidth=3, label="Real Electromagnetic Composites")
plt.xlabel('Frequency (GHz)')
plt.ylabel('Reflection Coefficient (dB)')
plt.title('Reflection Coefficient vs. Frequency')
plt.xscale('log')  # Logarithmic x-axis
plt.legend()
plt.grid(True, which="both", ls="--")  # Grid for both major and minor ticks

# Use ScalarFormatter for normal numbers instead of scientific notation
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())  # Hide minor tick labels

# Reduce number of tick labels
plt.xticks([.1, .2, .5, 1, 2, 5, 10])  # Adjust number of labels

# Rotate tick labels for better visibility
plt.xticks(rotation=30, ha="right")

plt.show()
