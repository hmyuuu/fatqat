"""Prepare and measure a Bell state

Follow a two-qubit Bell state from exact amplitudes to seeded measurement counts and a comparison with the ideal distribution.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

np.set_printoptions(precision=3, suppress=True)

# %%
bell_program = fq.Program(2)
bell_program.add(ops.H, 0)
bell_program.add(ops.CX, (0, 1))

# %%
backend = fq.simulator.Simulator(method="SV")
state_result = backend.run(
    bell_program,
    result_config={"counts": False, "final_state": True},
).result()
statevector = state_result.get_statevector()

print("Statevector:")
print(statevector)
print(f"Total probability: {np.vdot(statevector, statevector).real:.12f}")

# %%
probabilities = np.abs(statevector) ** 2
print("Exact basis probabilities:", probabilities)

# %%
measured_program = fq.Program(2, 2)
measured_program.add(ops.H, 0)
measured_program.add(ops.CX, (0, 1))
measured_program.measure((0, 1), (0, 1))

# %%
shots = 1_000
sample_result = backend.run(
    measured_program,
    shots=shots,
    simulation_config={"seed": 7},
).result()
counts = sample_result.get_counts()

print("Seeded counts:", counts)
print("Available result data:", sorted(sample_result.available_data))

# %%
outcomes = ("00", "01", "10", "11")
observed = np.array([counts.get(outcome, 0) / shots for outcome in outcomes])
ideal = np.array([0.5, 0.0, 0.0, 0.5])

observed_by_outcome = {
    outcome: float(frequency) for outcome, frequency in zip(outcomes, observed)
}
print("Observed frequencies:", observed_by_outcome)

figure, axis = plt.subplots(figsize=(7, 4))
positions = np.arange(len(outcomes))
axis.bar(positions, observed, width=0.65, label="seeded simulation")
axis.scatter(positions, ideal, color="black", marker="_", s=350, label="ideal")
axis.set(
    xticks=positions,
    xticklabels=outcomes,
    xlabel="Measured bitstring",
    ylabel="Frequency",
    ylim=(0, 0.6),
    title="Bell-state measurement frequencies",
)
axis.legend()
figure.tight_layout()
plt.show()
