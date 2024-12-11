import jax.numpy as jnp
import numpy as np
import scipy.constants as scic

from jaxlayerlumos import utils_materials
from jaxlayerlumos import stackrt_eps_mu

import matplotlib.pyplot as plt


def test_stackrt_radar():
    frequencies = jnp.linspace(0.1e9, 1e9, 3)  # in GHz
    # material_stack = jnp.array([3])
    # eps_stack, mu_stack = utils_materials.get_eps_mu_Michielssen(material_stack, frequencies)
    # d_stack = jnp.array([2]) * 1e-3


    materials = ['Air', '11', '16', '7', '4', '4', 'PEC']
    eps_stack, mu_stack = utils_materials.get_eps_mu(materials, frequencies)

    # eps_stack, mu_stack = utils_materials.get_eps_mu_Michielssen(
    #     material_stack, frequencies
    # )
    d_stack = jnp.array([0, 0.7742, 0.8485, 1.4878, 1.9883, 1.9863, 0]) * 1e-3
    R_TE, T_TE, R_TM, T_TM = stackrt_eps_mu(
        eps_stack, mu_stack, d_stack, frequencies, 0.0, materials
    )

    R_avg = (R_TE + R_TM) / 2
    T_avg = (T_TE + T_TM) / 2

    R_db = 10 * jnp.log10(R_avg).squeeze()

    # plt.close('all')
    # plt.semilogx(frequencies/1e9, R_db, label=r'$rSlab\_TE\_db$')
    # plt.xlabel('Frequency (GHz)')
    # plt.ylabel('Reflection Coefficient (dB)')
    # plt.title('Reflection Coefficient vs Frequency')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    expected_R_db = np.array(
        [
            -33.872113339369314,
            -39.141066380663602,
            -38.794125046637177,
        ]
    )

    print("R_db")
    for elem in R_db:
        print(elem)
    print("T_avg")
    for elem in T_avg:
        for elem2 in elem:
            print(elem2)

    np.testing.assert_allclose(R_db, expected_R_db)
