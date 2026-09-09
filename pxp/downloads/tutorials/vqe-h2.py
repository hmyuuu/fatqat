"""Find the ground-state energy of H₂ with VQE

Run exact, finite-shot, and noisy VQE loops for molecular hydrogen and make the variational bound and sampling uncertainty explicit.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import fatqat as fq
import fatqat.operations as op

H2_TERMS = [
    # (Pauli string, coefficient); leftmost character acts on qubit 0.
    ("II", -1.052373245772859),
    ("IZ", +0.39793742484318045),
    ("ZI", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX", +0.18093119978423156),
]

PAULI = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]]),
}

# np.kron's first factor matches FATQAT public qubit 0.
H_MATRIX = sum(
    coeff * np.kron(PAULI[pauli[0]], PAULI[pauli[1]]) for pauli, coeff in H2_TERMS
)
eigenvalues = np.linalg.eigvalsh(H_MATRIX)
E0 = eigenvalues[0]
print("exact spectrum:", np.round(eigenvalues, 5))
print(f"ground-state energy E0 = {E0:.5f} Ha")

CHEMICAL_ACCURACY = 1.6e-3  # 1.6 mHa, the conventional accuracy target

# %%
OFFSET = H2_TERMS[0][1]  # the II coefficient
COEFFS = np.array([coeff for pauli, coeff in H2_TERMS[1:]])

# %%
NUM_QUBITS = 2
NUM_ROUNDS = 2

OBSERVABLES = [
    fq.Observable.from_sparse([("Z", (1,), 1.0)], num_qubits=NUM_QUBITS),    # IZ
    fq.Observable.from_sparse([("Z", (0,), 1.0)], num_qubits=NUM_QUBITS),    # ZI
    fq.Observable.from_sparse([("ZZ", (0, 1), 1.0)], num_qubits=NUM_QUBITS),
    fq.Observable.from_sparse([("XX", (0, 1), 1.0)], num_qubits=NUM_QUBITS),
]

# %%
THETA = fq.ParameterVector("theta", 4)
def build_template():
    program = fq.Program(NUM_QUBITS)
    for r in range(NUM_ROUNDS):
        for q in range(NUM_QUBITS):
            program.add(op.RY(THETA[r * NUM_QUBITS + q]), q)
        program.add(op.CX, (0, 1))
    return program

template = build_template()

# %%
figure = template.draw("matplotlib")
figure.set_size_inches(10, 3)

# %%
ESTIMATOR_SV = fq.Estimator(fq.simulator.Simulator(method="SV"))


def energy_exact(theta):
    """Exact energy of the ansatz at ``theta``"""
    bound = template.assign_parameters({THETA: theta})
    expectations = ESTIMATOR_SV.run(bound, OBSERVABLES).result().get_expectation()
    return float(OFFSET + COEFFS @ expectations)

# %%
rng = np.random.default_rng(0)
x0 = rng.uniform(-0.1, 0.1, 4)


def _trace(energy, theta, trace):
    value = energy(theta)
    trace.append(value)
    if len(trace) % 25 == 0:
        print(f"eval {len(trace):4d}  energy {value:.5f}")
    return value


trace_exact = []
result_exact = minimize(
    lambda theta: _trace(energy_exact, theta, trace_exact),
    x0,
    method="COBYLA",
    options={"maxiter": 200, "rhobeg": 0.5},
)
print(f"exact VQE minimum {result_exact.fun:.5f} Ha "
      f"(error {result_exact.fun - E0:+.5f} Ha)")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(E0, color="k", ls="--", lw=1, label=f"exact $E_0$ = {E0:.4f}")
ax.axhspan(E0, E0 + CHEMICAL_ACCURACY, color="tab:green", alpha=0.2,
           label="chemical accuracy (1.6 mHa)")
ax.plot(trace_exact, label="exact VQE trace")
ax.set_xlabel("energy evaluation")
ax.set_ylabel("energy (Ha)")
ax.set_title("Exact VQE converges to the ground state")
ax.legend()
fig.tight_layout()

# %%
def energy_sampled(theta):
    """Finite-shot energy of the ansatz at ``theta``."""
    bound = template.assign_parameters({THETA: theta})
    expectations = (
        ESTIMATOR_SV.run(
            bound, OBSERVABLES, shots=1024, simulation_config={"seed": 7}
        )
        .result()
        .get_expectation()
    )
    return float(OFFSET + COEFFS @ expectations)


def sampled_std(theta):
    """Standard error of the finite-shot energy."""
    bound = template.assign_parameters({THETA: theta})
    std = (
        ESTIMATOR_SV.run(
            bound, OBSERVABLES, shots=1024, simulation_config={"seed": 7}
        )
        .result()
        .get_standard_error()
    )
    return float(np.sqrt(COEFFS**2 @ std**2))

# %%
trace_sampled = []
result_sampled = minimize(
    lambda theta: _trace(energy_sampled, theta, trace_sampled),
    x0,
    method="COBYLA",
    options={"maxiter": 200, "rhobeg": 0.5},
)
final_exact = energy_exact(result_sampled.x)
final_std = sampled_std(result_sampled.x)
print(f"finite-shot VQE stopped at {result_sampled.fun:.5f} ± {final_std:.5f} Ha")
print(f"exact energy at that point: {final_exact:.5f} Ha "
      f"(error {final_exact - E0:+.5f} Ha)")

# %%
fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4))
ax.axhline(E0, color="k", ls="--", lw=1, label=f"exact $E_0$ = {E0:.4f}")
ax.plot(trace_exact, label="exact objective")
ax.plot(trace_sampled, alpha=0.8, label="finite-shot objective (1024 shots)")
ax.set_xlabel("energy evaluation")
ax.set_ylabel("energy (Ha)")
ax.set_title("Finite-shot VQE: full traces")
ax.legend()

cut = 20  # skip the initial transient
ax_zoom.axhline(E0, color="k", ls="--", lw=1, label=f"exact $E_0$ = {E0:.4f}")
ax_zoom.axhspan(E0, E0 + CHEMICAL_ACCURACY, color="tab:green", alpha=0.2,
                label="chemical accuracy (1.6 mHa)")
ax_zoom.plot(range(cut, len(trace_exact)), trace_exact[cut:],
             label="exact objective")
ax_zoom.plot(range(cut, len(trace_sampled)), trace_sampled[cut:],
             alpha=0.8, marker=".", ms=4, label="finite-shot objective")
ax_zoom.errorbar(
    len(trace_sampled) - 1,
    trace_sampled[-1],
    yerr=final_std,
    fmt="o",
    color="tab:orange",
    capsize=4,
    label="standard error",
)
ax_zoom.set_xlabel("energy evaluation")
ax_zoom.set_title("zoom: riding the statistical noise")
ax_zoom.legend()
fig.tight_layout()

# %%
noise = fq.NoiseModel()
noise.add(fq.noise.Depolarizing(p=0.01), operation=op.RY)
noise.add(fq.noise.Depolarizing(p=0.05), operation=op.CX)
ESTIMATOR_NOISY = fq.Estimator(fq.simulator.Simulator(method="DM", noise=noise))


def energy_noisy(theta):
    """Noise-averaged energy of the ansatz at ``theta``."""
    bound = template.assign_parameters({THETA: theta})
    expectations = ESTIMATOR_NOISY.run(bound, OBSERVABLES).result().get_expectation()
    return float(OFFSET + COEFFS @ expectations)

# %%
degraded = energy_noisy(result_exact.x)
print(f"noiseless minimum under noise: {degraded:.5f} Ha "
      f"(shift {degraded - result_exact.fun:+.5f} Ha)")

trace_noisy = []
result_noisy = minimize(
    lambda theta: _trace(energy_noisy, theta, trace_noisy),
    x0,
    method="COBYLA",
    options={"maxiter": 200, "rhobeg": 0.5},
)
print(f"noisy VQE minimum {result_noisy.fun:.5f} Ha "
      f"(error vs. E0 {result_noisy.fun - E0:+.5f} Ha)")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.axhline(E0, color="k", ls="--", lw=1, label=f"exact $E_0$ = {E0:.4f}")
ax.plot(trace_exact, label="noiseless objective")
ax.plot(trace_noisy, label="noisy objective (depolarizing)")
ax.axhline(result_noisy.fun, color="tab:orange", ls=":", lw=1,
           label=f"noisy floor = {result_noisy.fun:.4f}")
ax.set_xlabel("energy evaluation")
ax.set_ylabel("energy (Ha)")
ax.set_title("Noise lifts the variational floor")
ax.legend()
fig.tight_layout()
