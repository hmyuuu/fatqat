"""Build antiferromagnetic correlations in a Rydberg chain

Design a three-stage Rydberg pulse from physical units and watch short-range antiferromagnetic order emerge in a ten-site chain.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

model_document = fq.emulator.load_model_document("atom2level.reference")
model = fq.emulator.Atom2LevelModel.from_document(model_document)

print("Model:", model_document["model"]["id"])
print("Basis:", model_document["system"]["basis"])
print("C6 unit:", model_document["units"]["c6"])

# %%
NUM_SITES = 10
OMEGA_MAX = 2 * np.pi  # rad/us
U = 2 * OMEGA_MAX  # nearest-pair Rydberg interaction, rad/us
J_ZZ = U / 4
C6 = model_document["parameters"]["c6"]
SPACING = (C6 / U) ** (1 / 6)

arrangement = fq.emulator.AtomArrangement.chain(
    num_sites=NUM_SITES,
    spacing=SPACING,
)

print(f"Nearest-pair U / 2pi = {U / (2 * np.pi):.3f} MHz")
print(f"Transverse h_x / 2pi = {OMEGA_MAX / (4 * np.pi):.3f} MHz")
print(f"Pauli J_ZZ / 2pi = {J_ZZ / (2 * np.pi):.3f} MHz")
print(f"Derived spacing = {SPACING:.3f} um")

# %%
DELTA_INITIAL = -1.5 * U
DELTA_FINAL = U / 3

# %%
T_RISE = 0.5
T_SWEEP = 1.0
T_FALL = 0.5


def pulse_stage(
    duration: float,
    omega: tuple[float, float],
    detuning: tuple[float, float],
) -> ops.PulseOperation:
    """Create one linear drive-and-detuning stage in model time units."""
    times = (0.0, duration)
    controls = (
        fq.emulator.PulseControl(
            model.control.drive(),
            fq.emulator.SampledWaveform(times, omega),
        ),
        fq.emulator.PulseControl(
            model.control.detuning(),
            fq.emulator.SampledWaveform(times, detuning),
        ),
    )
    return ops.PulseOperation(duration, controls)


program = fq.Program(arrangement.num_sites)
program.add(
    pulse_stage(
        T_RISE,
        omega=(0.0, OMEGA_MAX),
        detuning=(DELTA_INITIAL, DELTA_INITIAL),
    )
)
program.add(
    pulse_stage(
        T_SWEEP,
        omega=(OMEGA_MAX, OMEGA_MAX),
        detuning=(DELTA_INITIAL, DELTA_FINAL),
    )
)
program.add(
    pulse_stage(
        T_FALL,
        omega=(OMEGA_MAX, 0.0),
        detuning=(DELTA_FINAL, DELTA_FINAL),
    )
)

# %%
stage_boundaries = np.cumsum((0.0, T_RISE, T_SWEEP, T_FALL))
omega_nodes = np.array((0.0, OMEGA_MAX, OMEGA_MAX, 0.0))
detuning_nodes = np.array((DELTA_INITIAL, DELTA_INITIAL, DELTA_FINAL, DELTA_FINAL))

figure, axis = plt.subplots(figsize=(7, 4))
axis.plot(stage_boundaries, omega_nodes / U, marker="o", label=r"$\Omega/U$")
axis.plot(stage_boundaries, detuning_nodes / U, marker="o", label=r"$\Delta/U$")
for boundary in stage_boundaries[1:-1]:
    axis.axvline(boundary, color="0.8", linestyle="--", linewidth=1)
axis.axhline(0.0, color="0.25", linewidth=0.8)
axis.set(
    xlabel="Time (us)",
    ylabel="Control / U",
    title="Rise, detuning sweep, and fall",
)
axis.legend()
figure.tight_layout()
plt.show()

# %%
site_occupations = [
    fq.Observable.from_sparse(
        [("ONE", (site,), 1.0)],
        num_qubits=NUM_SITES,
    )
    for site in range(NUM_SITES)
]
pair_indices = [
    (first, second)
    for first in range(NUM_SITES)
    for second in range(first + 1, NUM_SITES)
]
pair_occupations = [
    fq.Observable.from_sparse(
        [(["ONE", "ONE"], pair, 1.0)],
        num_qubits=NUM_SITES,
    )
    for pair in pair_indices
]
staggered_order_squared = fq.Observable.from_sparse(
    [
        (
            "I",
            (0,),
            1 / NUM_SITES,
        )
    ]
    + [
        (
            "ZZ",
            (first, second),
            2 * (-1.0) ** (first + second) / NUM_SITES**2,
        )
        for first in range(NUM_SITES)
        for second in range(first + 1, NUM_SITES)
    ],
    num_qubits=NUM_SITES,
)
observables = site_occupations + pair_occupations + [staggered_order_squared]

# %%
backend = fq.emulator.Atom2LevelEmulator(
    model,
    arrangement=arrangement,
)
estimator = fq.Estimator(backend)
run_configs = {
    "all pairs": {"interaction_cutoff": None},
    "nearest-pair cutoff": {"interaction_cutoff": SPACING},
}

site_results = {}
double_results = {}
staggered_results = {}
connected_results = {}

for label, simulation_config in run_configs.items():
    values = np.asarray(
        estimator.run(
            program,
            observables,
            simulation_config=simulation_config,
        )
        .result()
        .get_expectation()
    )
    occupations = values[:NUM_SITES]
    pair_values = values[NUM_SITES:-1]
    pair_lookup = dict(zip(pair_indices, pair_values))
    site_results[label] = occupations
    double_results[label] = float(
        np.mean([pair_lookup[(site, site + 1)] for site in range(NUM_SITES - 1)])
    )
    staggered_results[label] = float(values[-1])
    connected = np.diag(occupations * (1 - occupations))
    for (first, second), pair_value in pair_lookup.items():
        correlation = pair_value - occupations[first] * occupations[second]
        connected[first, second] = correlation
        connected[second, first] = correlation
    connected_results[label] = connected
    print(
        f"{label:>19}: "
        f"squared staggered order = {staggered_results[label]:.3f}, "
        f"adjacent double excitation = {double_results[label]:.3f}"
    )

# %%
sites = np.arange(NUM_SITES)
bar_width = 0.36
figure, (density_axis, correlation_axis, summary_axis) = plt.subplots(
    1, 3, figsize=(15, 4)
)

for index, (label, occupations) in enumerate(site_results.items()):
    offset = (index - 0.5) * bar_width
    density_axis.bar(sites + offset, occupations, bar_width, label=label)

density_axis.set(
    xlabel="Site",
    ylabel=r"Rydberg population $\langle n_i\rangle$",
    xticks=sites,
    ylim=(0.0, 1.08),
    title="Finite-chain populations",
)
density_axis.legend(fontsize="small")

physical_correlations = connected_results["all pairs"]
color_bound = np.max(np.abs(physical_correlations))
image = correlation_axis.imshow(
    physical_correlations,
    cmap="RdBu_r",
    vmin=-color_bound,
    vmax=color_bound,
)
correlation_axis.set(
    xlabel="Site j",
    ylabel="Site i",
    xticks=sites,
    yticks=sites,
    title=r"All-pair connected $C_{ij}$",
)
figure.colorbar(image, ax=correlation_axis, fraction=0.046, pad=0.04)

metric_names = ("squared staggered\norder", "adjacent double\nexcitation")
metric_positions = np.arange(len(metric_names))
for index, label in enumerate(run_configs):
    offset = (index - 0.5) * bar_width
    bars = summary_axis.bar(
        metric_positions + offset,
        (staggered_results[label], double_results[label]),
        bar_width,
        label=label,
    )
    summary_axis.bar_label(bars, fmt="%.3f", padding=2, fontsize="small")

summary_axis.set(
    ylabel="Expectation value",
    xticks=metric_positions,
    xticklabels=metric_names,
    ylim=(0.0, 0.5),
    title="Order and blockade",
)
summary_axis.legend(fontsize="small")
figure.tight_layout()
plt.show()
