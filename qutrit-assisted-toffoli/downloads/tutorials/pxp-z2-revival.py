"""Revivals and entanglement growth in an open PXP chain

Trotterize the constrained PXP Hamiltonian and compare many-body revivals and half-chain entropy with an independent exact solve.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

import fatqat as fq

NUM_SITES = 10
HALF = NUM_SITES // 2
DIM = 2**NUM_SITES

OMEGA = 2 * np.pi  # rad/us -> PXP coefficient g = OMEGA / 2
T_MAX = 6.0  # us, covers the first three revivals
DT_TROTTER = 0.01  # us, one symmetric second-order Trotter step
TIME_GRID = np.linspace(0.0, T_MAX, 121)

Z2_BITS = tuple(1 - (i % 2) for i in range(NUM_SITES))  # |r g r g ...>
Z2_INDEX = sum(2 ** (NUM_SITES - 1 - i) for i in range(NUM_SITES) if Z2_BITS[i])
Z2 = np.zeros(DIM, dtype=complex)
Z2[Z2_INDEX] = 1.0

ALT_BITS = tuple(1 - bit for bit in Z2_BITS)  # the twin Neel branch
ALT_INDEX = sum(2 ** (NUM_SITES - 1 - i) for i in range(NUM_SITES) if ALT_BITS[i])
ALT = np.zeros(DIM, dtype=complex)
ALT[ALT_INDEX] = 1.0

print(f"Z2 branch |r g r g ...> sits at statevector index {Z2_INDEX}")
print(f"twin branch |g r g r ...> sits at statevector index {ALT_INDEX}")

# %%
from dataclasses import dataclass
from typing import ClassVar

from fatqat.operations import Operation

THETA = OMEGA * DT_TROTTER / 4


@dataclass(frozen=True)
class PXPBulk(Operation):
    """Three-site PXP exponential: X on the middle site guarded by two P's."""

    name: ClassVar[str] = "PXPBulk"
    num_subsystems: ClassVar[int] = 3


@dataclass(frozen=True)
class PXPEdgeLeft(Operation):
    """Left boundary term X_0 P_1."""

    name: ClassVar[str] = "PXPEdgeLeft"
    num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class PXPEdgeRight(Operation):
    """Right boundary term P_{L-2} X_{L-1}."""

    name: ClassVar[str] = "PXPEdgeRight"
    num_subsystems: ClassVar[int] = 2


def _rotation_matrix(dimension: int, pair: tuple[int, int], angle: float) -> np.ndarray:
    """Identity except for a 2x2 exp(-i angle X) rotation on ``pair``."""
    matrix = np.eye(dimension, dtype=complex)
    first, second = pair
    matrix[first, first] = matrix[second, second] = np.cos(angle)
    matrix[first, second] = matrix[second, first] = -1j * np.sin(angle)
    return matrix


# fatqat flattens local matrices with the FIRST target as the most
# significant bit: index = b_{t0} * 2**(k-1) + ... + b_{t_{k-1}}.
# Keep that in mind or the edge terms quietly act on the wrong site.
# Bulk (i-1, i, i+1): flip the middle site -> pair (|ggg>, |grg>) = (0, 2).
BULK_MATRIX = _rotation_matrix(8, (0, 2), THETA)
# Left edge (0, 1): flip site 0 -> pair (|gg>, |rg>) = (0, 2).
EDGE_LEFT_MATRIX = _rotation_matrix(4, (0, 2), THETA)
# Right edge (L-2, L-1): flip site L-1 -> pair (|gg>, |gr>) = (0, 1).
EDGE_RIGHT_MATRIX = _rotation_matrix(4, (0, 1), THETA)


implementation_map = fq.implementation.MatrixImplementationMap()
implementation_map.add(PXPBulk, BULK_MATRIX)
implementation_map.add(PXPEdgeLeft, EDGE_LEFT_MATRIX)
implementation_map.add(PXPEdgeRight, EDGE_RIGHT_MATRIX)

# NumPy runtime on purpose: the program is thousands of tiny local
# operations, and at that size the Python loop dwarfs any matrix kernel,
# so numba would only add compile time.
backend = fq.simulator.Simulator(
    method="SV",
    runtime="numpy",
    implementation_map=implementation_map,
)


def trotter_program(duration: float) -> fq.Program:
    """Build the symmetric second-order Trotter program for ``duration``."""
    steps = int(round(duration / DT_TROTTER))
    program = fq.Program(NUM_SITES)
    for _ in range(steps):
        for site in range(NUM_SITES):  # forward half pass
            if site == 0:
                program.add(PXPEdgeLeft(), (0, 1))
            elif site == NUM_SITES - 1:
                program.add(PXPEdgeRight(), (NUM_SITES - 2, NUM_SITES - 1))
            else:
                program.add(PXPBulk(), (site - 1, site, site + 1))
        for site in range(NUM_SITES - 1, -1, -1):  # backward half pass
            if site == 0:
                program.add(PXPEdgeLeft(), (0, 1))
            elif site == NUM_SITES - 1:
                program.add(PXPEdgeRight(), (NUM_SITES - 2, NUM_SITES - 1))
            else:
                program.add(PXPBulk(), (site - 1, site, site + 1))
    return program


def evolve_fatqat(duration: float) -> np.ndarray:
    """Quench |Z2> for ``duration`` and return the final statevector."""
    if duration == 0.0:
        return Z2.copy()
    result = backend.run(
        trotter_program(duration),
        initial_state=Z2,
        result_config={"counts": False, "final_state": True},
    ).result()
    return result.get_statevector()

# %%
import qutip

_P = (qutip.qeye(2) + qutip.sigmaz()) / 2  # |g><g|
_X = qutip.sigmax()
_I2 = qutip.qeye(2)


def _pxp_term(site: int) -> "qutip.Qobj":
    factors = [_I2] * NUM_SITES
    factors[site] = _X
    if site - 1 >= 0:
        factors[site - 1] = _P
    if site + 1 < NUM_SITES:
        factors[site + 1] = _P
    return qutip.tensor(*factors)


ORACLE_H = sum((OMEGA / 2) * _pxp_term(site) for site in range(NUM_SITES))
ORACLE_Z2 = qutip.tensor(*[qutip.basis(2, bit) for bit in Z2_BITS])
ORACLE_RESULT = qutip.sesolve(ORACLE_H, ORACLE_Z2, list(TIME_GRID))

def evolve_oracle(index: int) -> np.ndarray:
    """Return one QuTiP statevector."""
    return np.asarray(ORACLE_RESULT.states[index].full()).reshape(-1)

# %%
def fidelity(state: np.ndarray, reference: np.ndarray) -> float:
    """Return |<reference|state>|^2 for two statevectors."""
    return float(abs(np.vdot(reference, state)) ** 2)


def half_chain_entropy(state: np.ndarray) -> float:
    """Von Neumann entropy of the first-half subsystem, via the SVD."""
    schmidt = np.linalg.svd(state.reshape(2**HALF, 2**HALF), compute_uv=False)
    probabilities = schmidt**2
    probabilities = probabilities[probabilities > 0.0]
    return -float(np.sum(probabilities * np.log(probabilities)))


def site_occupations(state: np.ndarray) -> np.ndarray:
    """Marginal |r> population of every site from one statevector."""
    basis_indices = np.arange(DIM)
    return np.array(
        [
            float(
                np.sum(
                    np.abs(state) ** 2
                    * ((basis_indices >> (NUM_SITES - 1 - site)) & 1)
                )
            )
            for site in range(NUM_SITES)
        ]
    )

# %%
fatqat_states = [evolve_fatqat(t) for t in TIME_GRID]

fatqat_fidelity = np.array([fidelity(s, Z2) for s in fatqat_states])
fatqat_alt = np.array([fidelity(s, ALT) for s in fatqat_states])
fatqat_entropy = np.array([half_chain_entropy(s) for s in fatqat_states])
fatqat_occupations = np.array([site_occupations(s) for s in fatqat_states])

oracle_fidelity = np.array(
    [fidelity(evolve_oracle(i), Z2) for i in range(len(TIME_GRID))]
)
oracle_entropy = np.array(
    [half_chain_entropy(evolve_oracle(i)) for i in range(len(TIME_GRID))]
)

max_gap = float(np.max(np.abs(fatqat_fidelity - oracle_fidelity)))
print(f"Largest fidelity gap between Trotter and oracle: {max_gap:.5f}")

# %%
peaks = argrelextrema(fatqat_fidelity, np.greater, order=4)[0]
peaks = [p for p in peaks if TIME_GRID[p] > 0.2 and fatqat_fidelity[p] > 0.2]

print("\nRevivals of the |Z2> quench (fatqat Trotter):")
print(f"{'time (us)':>10} {'F(Z2)':>8} {'F(alt)':>8} {'S(t)':>8}")
for p in peaks[:4]:
    print(
        f"{TIME_GRID[p]:>10.2f} {fatqat_fidelity[p]:>8.3f} "
        f"{fatqat_alt[p]:>8.3f} {fatqat_entropy[p]:>8.3f}"
    )

first_peak_time = TIME_GRID[peaks[0]]
first_peak_fidelity = fatqat_fidelity[peaks[0]]
first_peak_entropy = fatqat_entropy[peaks[0]]
print(f"Entropy maximum: {fatqat_entropy.max():.3f}")

# %%
figure, (fid_axis, ent_axis, occ_axis) = plt.subplots(1, 3, figsize=(16, 4))

fid_axis.plot(
    TIME_GRID,
    fatqat_fidelity,
    label="fatqat Trotter",
    linewidth=2,
)
fid_axis.plot(TIME_GRID, oracle_fidelity, "k--", label="exact PXP oracle")
fid_axis.scatter(
    TIME_GRID[peaks],
    fatqat_fidelity[peaks],
    color="C0",
    marker="o",
    zorder=3,
    label="revival peaks",
)
fid_axis.set(
    xlabel="Time (us)",
    ylabel=r"Fidelity $|\langle Z_2|\psi(t)\rangle|^2$",
    ylim=(0, 1.05),
    title="Z2 revival",
)
fid_axis.legend(fontsize="small")

ent_axis.plot(TIME_GRID, fatqat_entropy, label="fatqat Trotter", linewidth=2)
ent_axis.plot(TIME_GRID, oracle_entropy, "k--", label="exact PXP oracle")
for p in peaks[:3]:
    ent_axis.axvline(TIME_GRID[p], color="0.8", linestyle=":", linewidth=1)
ent_axis.set(
    xlabel="Time (us)",
    ylabel=r"Half-chain entropy $S(t)$",
    title="Entanglement growth",
)
ent_axis.legend(fontsize="small")

image = occ_axis.imshow(
    fatqat_occupations.T,
    aspect="auto",
    origin="lower",
    extent=(TIME_GRID[0], TIME_GRID[-1], 0, NUM_SITES),
    cmap="viridis",
)
occ_axis.set(
    xlabel="Time (us)",
    ylabel="Site",
    title=r"Rydberg population $\langle n_i(t)\rangle$",
)
figure.colorbar(image, ax=occ_axis, fraction=0.046, pad=0.04)
figure.tight_layout()
plt.show()
