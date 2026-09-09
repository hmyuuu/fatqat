"""Solve a QUBO with QAOA

Map a constrained combinatorial problem to a QUBO, turn it into an Ising Hamiltonian and a FatQat program, and read the answer back out of the measured distribution.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import fatqat as fq
import fatqat.operations as ops

# The WSCC 9-bus network. Node labels are zero-based, so node 0 is bus 1.
EDGES = [
    # (node, node, weight)
    (0, 3, 0.8855058426108287),
    (2, 5, 1.0506270719315443),
    (3, 4, 0.3795071371756076),
    (4, 5, 0.7349783693597357),
    (5, 6, 0.29891469957856975),
    (6, 7, 0.9382048153285490),
    (7, 1, 2.0147319144099027),
    (7, 8, 1.0706524421112620),
    (8, 3, 0.5028157475281494),
]
NODES = sorted({node for i, j, _ in EDGES for node in (i, j)})
N = len(NODES)
SEPARATE = (1, 7)  # these two nodes must land on opposite sides

BALANCE_WEIGHT = 1.0
SEPARATE_WEIGHT = 4.0

print(f"{N} nodes, {len(EDGES)} edges, total edge weight {sum(w for *_, w in EDGES):.3f}")

# %%
constant = 0.0
linear = np.zeros(N)
quadratic: dict[tuple[int, int], float] = {}


def add_quadratic(i, j, value):
    """Accumulate a coefficient on the pair (i, j), stored with i < j."""
    key = (i, j) if i < j else (j, i)
    quadratic[key] = quadratic.get(key, 0.0) + value


# Objective: the weight of every cut edge.
for i, j, weight in EDGES:
    linear[i] += weight
    linear[j] += weight
    add_quadratic(i, j, -2.0 * weight)

# Constraint: equal halves, as (sum_i x_i - N/2)**2 expanded with x*x = x.
target = N / 2
constant += BALANCE_WEIGHT * target**2
for i in NODES:
    linear[i] += BALANCE_WEIGHT * (1.0 - 2.0 * target)
for i in NODES:
    for j in NODES:
        if i < j:
            add_quadratic(i, j, 2.0 * BALANCE_WEIGHT)

# Constraint: nodes 0 and 5 on opposite sides.
first, second = SEPARATE
constant += SEPARATE_WEIGHT
linear[first] -= SEPARATE_WEIGHT
linear[second] -= SEPARATE_WEIGHT
add_quadratic(first, second, 2.0 * SEPARATE_WEIGHT)

print(f"constant {constant:.3f}")
print(f"linear   {np.round(linear, 3)}")
print(f"{len(quadratic)} pair terms")

# %%
codes = np.arange(1 << N)
bits = ((codes[:, None] >> np.arange(N - 1, -1, -1)[None, :]) & 1).astype(float)

spectrum = np.full(1 << N, constant)
spectrum += bits @ linear
for (i, j), value in quadratic.items():
    spectrum += value * bits[:, i] * bits[:, j]

optimal_indices = np.flatnonzero(np.isclose(spectrum, spectrum.min()))
uniform_probability = optimal_indices.size / spectrum.size

print(f"minimum energy {spectrum.min():.4f}")
for index in optimal_indices:
    assignment = format(index, f"0{N}b")
    side_one = [node for node in NODES if assignment[node] == "1"]
    side_zero = [node for node in NODES if assignment[node] == "0"]
    print(f"  {assignment}  side 0 = {side_zero}, side 1 = {side_one}")
print(f"{optimal_indices.size} optimal assignments out of {spectrum.size}")
print(f"a uniform sampler finds one with probability {uniform_probability:.4f}")

# %%
offset = constant + linear.sum() / 2 + sum(quadratic.values()) / 4
field = -linear / 2
for (i, j), value in quadratic.items():
    field[i] -= value / 4
    field[j] -= value / 4
coupling = {key: value / 4 for key, value in quadratic.items()}

# Summing many penalty terms leaves rounding residue where a coefficient should
# cancel exactly. Dropping it keeps those non-terms from becoming gates.
TOLERANCE = 1e-12
field[np.abs(field) <= TOLERANCE] = 0.0
coupling = {key: value for key, value in coupling.items() if abs(value) > TOLERANCE}

spins = 1 - 2 * bits
ising_energies = np.full(1 << N, offset) + spins @ field
for (i, j), value in coupling.items():
    ising_energies += value * spins[:, i] * spins[:, j]

print(f"offset {offset:.4f}")
print(f"field  {np.round(field, 12)}")
print(f"{len(coupling)} couplings, largest |J| = {max(abs(v) for v in coupling.values()):.4f}")
print(f"max |QUBO - Ising| over all {spectrum.size} assignments: "
      f"{np.abs(spectrum - ising_energies).max():.2e}")

# %%
def qaoa_program(betas, gammas, *, measure=False):
    """Build the QAOA program for this problem at the given angles."""
    program = fq.Program(N, N if measure else 0)
    for qubit in range(N):
        program.add(ops.H, qubit)

    for beta, gamma in zip(betas, gammas):
        for qubit, value in enumerate(field):
            if value != 0.0:
                program.add(ops.RZ(2.0 * gamma * value), qubit)
        for (i, j), value in coupling.items():
            program.add(ops.CX, (i, j))
            program.add(ops.RZ(2.0 * gamma * value), j)
            program.add(ops.CX, (i, j))
        for qubit in range(N):
            program.add(ops.RX(2.0 * beta), qubit)

    if measure:
        program.measure_all()
    return program


demo = qaoa_program([0.4], [0.15])

# Draw onto an axis this cell created, so the figure belongs to pyplot.
figure, axis = plt.subplots(figsize=(11.0, 3.4))
demo.draw(ax=axis)
axis.set_title("One QAOA layer")
figure.tight_layout()

print(f"depth-one program: {N} qubits, {len(coupling)} couplings, "
      f"{2 * len(coupling)} two-qubit gates per layer")

# %%
terms = [("I", (0,), offset)]
terms += [("Z", (index,), value) for index, value in enumerate(field) if value != 0.0]
terms += [("ZZ", (i, j), value) for (i, j), value in coupling.items()]
hamiltonian = fq.Observable.from_sparse(terms, num_qubits=N)

simulator = fq.simulator.Simulator(method="statevector")
estimator = fq.Estimator(simulator)


def probabilities(betas, gammas):
    """Return the exact output distribution over the 2**N bitstrings."""
    result = simulator.run(
        qaoa_program(betas, gammas), shots=1, result_config={"final_state": True}
    ).result()
    return np.abs(result.get_statevector()) ** 2


def energy(betas, gammas):
    """Return the exact mean energy, contracted against the cost spectrum."""
    return float(probabilities(betas, gammas) @ spectrum)


test_betas, test_gammas = [0.4], [0.15]
from_estimator = float(
    estimator.run(qaoa_program(test_betas, test_gammas), hamiltonian).result().get_expectation()
)
from_spectrum = energy(test_betas, test_gammas)

print(f"estimator            {from_estimator:.10f}")
print(f"distribution contract {from_spectrum:.10f}")
print(f"difference            {abs(from_estimator - from_spectrum):.2e}")

# %%
gamma_scale = max(abs(value) for value in coupling.values())
beta_grid = np.linspace(0.0, np.pi, 49)
gamma_grid = np.linspace(0.0, np.pi / gamma_scale, 97)
landscape = np.array([[energy([b], [g]) for g in gamma_grid] for b in beta_grid])

row, column = np.unravel_index(np.argmin(landscape), landscape.shape)
grid_best = (beta_grid[row], gamma_grid[column])

figure, axis = plt.subplots(figsize=(7.0, 3.4))
mesh = axis.pcolormesh(gamma_grid, beta_grid, landscape, shading="auto", cmap="viridis")
axis.plot(*grid_best[::-1], "w*", markersize=13, label="grid minimum")
axis.set_xlabel(r"$\gamma$")
axis.set_ylabel(r"$\beta$")
axis.set_title("Depth-one QAOA energy landscape")
axis.legend(loc="upper right")
figure.colorbar(mesh, ax=axis, label=r"$\langle H \rangle$")
figure.tight_layout()

print(f"largest coupling |J| = {gamma_scale:.3f}, so gamma is scanned over [0, {np.pi / gamma_scale:.3f})")
print(f"grid minimum {landscape.min():.4f} at beta={grid_best[0]:.3f}, gamma={grid_best[1]:.3f}")
print(f"random-guess average {landscape.mean():.4f}, exact minimum {spectrum.min():.4f}")

# %%
rng = np.random.default_rng(2024)
depths = [1, 2, 3, 4, 5]
mean_energies, optimum_probabilities = [], []
betas = gammas = None

for depth in depths:
    def objective(parameters, depth=depth):
        return energy(parameters[:depth], parameters[depth:])

    if betas is None:
        # Cold start: random guesses on the scales the landscape showed.
        starts = [
            np.concatenate(
                [rng.uniform(0, np.pi, depth), rng.uniform(0, np.pi / gamma_scale, depth)]
            )
            for _ in range(8)
        ]
    else:
        starts = [
            # Repeat the last layer, which usually lands in a deeper basin.
            np.concatenate([betas, betas[-1:], gammas, gammas[-1:]]),
            # Make the new layer the identity, so this depth starts exactly at
            # the previous optimum and can never do worse.
            np.concatenate([betas, [0.0], gammas, [0.0]]),
        ]
        starts += [
            np.concatenate(
                [betas, rng.uniform(0, np.pi, 1), gammas, rng.uniform(0, np.pi / gamma_scale, 1)]
            )
            for _ in range(2)
        ]

    best = min(
        (
            minimize(
                objective,
                start,
                method="COBYLA",
                options={"maxiter": 120, "rhobeg": np.pi / (4 * gamma_scale), "tol": 1e-8},
            )
            for start in starts
        ),
        key=lambda outcome: outcome.fun,
    )
    betas, gammas = best.x[:depth], best.x[depth:]

    distribution = probabilities(betas, gammas)
    mean_energies.append(float(best.fun))
    optimum_probabilities.append(float(distribution[optimal_indices].sum()))
    print(
        f"p={depth}: mean energy {mean_energies[-1]:7.4f}   "
        f"P(optimal) {optimum_probabilities[-1]:.4f}   "
        f"{optimum_probabilities[-1] / uniform_probability:5.1f}x uniform"
    )

print(f"exact minimum {spectrum.min():.4f}")

# %%
figure, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.2))

left.plot(depths, mean_energies, "o-", color="#1f77b4")
left.axhline(spectrum.min(), color="#444444", linestyle="--", label="exact minimum")
left.axhline(spectrum.mean(), color="#aaaaaa", linestyle=":", label="random guessing")
left.set_xlabel("QAOA depth $p$")
left.set_ylabel(r"$\langle H \rangle$")
left.set_title("Mean energy")
left.set_xticks(depths)
left.legend()

right.plot(depths, optimum_probabilities, "s-", color="#d62728")
right.axhline(uniform_probability, color="#aaaaaa", linestyle=":", label="uniform sampling")
right.set_xlabel("QAOA depth $p$")
right.set_ylabel("probability of an optimal assignment")
right.set_title("Solution probability")
right.set_xticks(depths)
right.legend()

figure.tight_layout()

# %%
SHOTS = 4000
measured = simulator.run(
    qaoa_program(betas, gammas, measure=True),
    shots=SHOTS,
    simulation_config={"seed": 11},
).result()
counts = measured.get_counts()

ranked = sorted(counts.items(), key=lambda item: spectrum[int(item[0], 2)])
print(f"{len(counts)} distinct outcomes in {SHOTS} shots")
print("best outcomes by true objective value:")
for assignment, shots in ranked[:4]:
    side_one = [node for node in NODES if assignment[node] == "1"]
    cut = sum(w for i, j, w in EDGES if (assignment[i] == "1") != (assignment[j] == "1"))
    print(
        f"  {assignment}  energy {spectrum[int(assignment, 2)]:7.4f}  "
        f"cut weight {cut:.2f}  side 1 = {side_one}  ({shots} shots)"
    )

best_assignment = ranked[0][0]
print(f"\nbest sampled energy {spectrum[int(best_assignment, 2)]:.4f}, "
      f"exact minimum {spectrum.min():.4f}")

# %%
figure, axis = plt.subplots(figsize=(8.0, 3.4))
measured.draw(
    number_to_keep=12,
    sort="count",
    ax=axis,
    title=f"Most frequent outcomes ({SHOTS} shots)",
)

optimal_strings = {format(index, f"0{N}b") for index in optimal_indices}
for label in axis.get_xticklabels():
    if label.get_text() in optimal_strings:
        label.set_color("#d62728")
        label.set_fontweight("bold")
axis.set_xlabel("outcome (optimal assignments in red)")
figure.tight_layout()
