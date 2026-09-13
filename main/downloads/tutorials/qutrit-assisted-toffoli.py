"""Build a Toffoli gate by borrowing a third level

Use a qubit–qutrit–qubit register to construct Toffoli with three qubit–qutrit interactions, then verify its relative phases and return from the borrowed level.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

a = fq.QuantumRegister(1, name="a")
b = fq.QuantumRegister(1, name="b", dim=3)
t = fq.QuantumRegister(1, name="t")
dims = (2, 3, 2)
basis_labels = ["".join(map(str, digits)) for digits in np.ndindex(dims)]

# %%
def basis_state(a_level, b_level, t_level):
    state = np.zeros(12, dtype=complex)
    state[6 * a_level + 2 * b_level + t_level] = 1
    return state

# %%
controlled_exchange = [
    (ops.SubspaceRY(np.pi / 2, (1, 2)), b[0]),
    (ops.CClock(1), (b[0], a[0])),
    (ops.SubspaceRY(-np.pi / 2, (1, 2)), b[0]),
]

# %%
flip_on_level_2 = [
    (ops.SwapLevels(1, 2), b[0]),
    (ops.H, t[0]),
    (ops.CClock(1), (b[0], t[0])),
    (ops.H, t[0]),
    (ops.SwapLevels(1, 2), b[0]),
]

# %%
sequence = controlled_exchange + flip_on_level_2 + controlled_exchange


def make_program(instructions):
    program = fq.Program([a, b, t])
    for operation, targets in instructions:
        program.add(operation, targets)
    return program


toffoli_program = make_program(sequence)

# %%
state_backend = fq.simulator.Simulator(method="statevector", runtime="numpy")


def evolve(instructions, initial_state):
    result = state_backend.run(
        make_program(instructions),
        initial_state=initial_state,
        result_config={"counts": False, "final_state": True},
    ).result()
    return result.get_statevector()

# %%
checkpoints = [0, 3, 8, 11]
stage_names = ["Input", "Exchange", "Flip target", "Restore"]
branch_states = [evolve(sequence[:stop], basis_state(1, 1, 0)) for stop in checkpoints]
expected_digits = [(1, 1, 0), (1, 2, 0), (1, 2, 1), (1, 1, 1)]

for state, digits in zip(branch_states, expected_digits):
    np.testing.assert_allclose(state, basis_state(*digits), atol=1e-12, rtol=0)

branch_labels = [basis_labels[np.argmax(np.abs(state))] for state in branch_states]
borrowed_populations = [
    np.sum(np.abs(state.reshape(dims)[:, 2, :]) ** 2) for state in branch_states
]

for name, label, population in zip(stage_names, branch_labels, borrowed_populations):
    print(f"{name:>11}: |{label}>, P(b=2) = {population:.3e}")

# %%
figure, axis = plt.subplots(figsize=(7.6, 4.2))
positions = np.arange(len(checkpoints))
axis.bar(positions, borrowed_populations, width=0.5, color="#c97920")
axis.scatter(positions, borrowed_populations, color="#805014", zorder=3)
for position, label in zip(positions, branch_labels):
    axis.text(
        position, 1.19, rf"$|{label}\rangle$", ha="center", va="center", fontsize=16
    )
for position in positions[:-1]:
    axis.annotate(
        "",
        xy=(position + 0.72, 1.19),
        xytext=(position + 0.28, 1.19),
        arrowprops={"arrowstyle": "->", "color": "0.45"},
    )
axis.set(
    xticks=positions,
    xticklabels=stage_names,
    yticks=[0, 0.5, 1],
    ylim=(-0.05, 1.4),
    xlabel="Circuit checkpoint",
    ylabel=r"Borrowed-level population $P(b=2)$",
    title=r"Borrow, flip, return: input $|110\rangle$",
)
axis.spines[["top", "right"]].set_visible(False)
figure.tight_layout()
plt.show()

# %%
preparation = [
    (ops.H, a[0]),
    (ops.S, a[0]),
    (ops.Shift(1), b[0]),
]
coherent_input = evolve(preparation, basis_state(0, 0, 0))
expected_input = (basis_state(0, 1, 0) + 1j * basis_state(1, 1, 0)) / np.sqrt(2)
np.testing.assert_allclose(coherent_input, expected_input, atol=1e-12, rtol=0)

coherent_output = evolve(sequence, coherent_input)
expected_output = (basis_state(0, 1, 0) + 1j * basis_state(1, 1, 1)) / np.sqrt(2)
np.testing.assert_allclose(coherent_output, expected_output, atol=1e-12, rtol=0)
np.testing.assert_allclose(
    coherent_output.reshape(dims)[:, 2, :], 0, atol=1e-12, rtol=0
)

print("Nonzero output amplitudes:")
for label, amplitude in zip(basis_labels, coherent_output):
    if abs(amplitude) > 1e-12:
        print(f"  |{label}>: {amplitude:.6f}")
print(
    "Final borrowed-level population:",
    np.sum(np.abs(coherent_output.reshape(dims)[:, 2, :]) ** 2),
)

# %%
unitary = (
    fq.simulator.Simulator(method="unitary", runtime="numpy")
    .run(toffoli_program, result_config={"counts": False, "final_state": True})
    .result()
    .get_unitary()
)

logical_indices = np.array([0, 1, 2, 3, 6, 7, 8, 9])
borrowed_indices = np.array([4, 5, 10, 11])
expected_ccx = np.zeros((8, 8), dtype=complex)
for column, (a_bit, b_bit, t_bit) in enumerate(np.ndindex(2, 2, 2)):
    output_bit = t_bit ^ (a_bit & b_bit)
    row = 4 * a_bit + 2 * b_bit + output_bit
    expected_ccx[row, column] = 1

logical_unitary = unitary[np.ix_(logical_indices, logical_indices)]
leakage_block = unitary[np.ix_(borrowed_indices, logical_indices)]
np.testing.assert_allclose(logical_unitary, expected_ccx, atol=1e-12, rtol=0)
np.testing.assert_allclose(leakage_block, 0, atol=1e-12, rtol=0)

print("Maximum complex logical error:", np.max(np.abs(logical_unitary - expected_ccx)))
print("Maximum final leakage amplitude:", np.max(np.abs(leakage_block)))

# %%
transition_probabilities = np.abs(unitary[:, logical_indices]) ** 2
logical_labels = [basis_labels[index] for index in logical_indices]

figure, axis = plt.subplots(figsize=(7.2, 6.3))
heatmap = axis.imshow(transition_probabilities, cmap="Blues", vmin=0, vmax=1)
axis.set(
    xticks=np.arange(8),
    xticklabels=[rf"$|{label}\rangle$" for label in logical_labels],
    yticks=np.arange(12),
    yticklabels=[rf"$|{label}\rangle$" for label in basis_labels],
    xlabel="Logical input",
    ylabel="Output in the (2, 3, 2) register",
    title="Toffoli action in the larger state space",
)
for row in borrowed_indices:
    axis.axhspan(
        row - 0.5, row + 0.5, facecolor="none", edgecolor="#c97920", linewidth=1.5
    )
    axis.get_yticklabels()[row].set_color("#996019")
figure.colorbar(heatmap, ax=axis, label="Transition probability", shrink=0.8)
figure.tight_layout()
plt.show()

# %%
interaction_count = sum(operation.num_subsystems == 2 for operation, _ in sequence)
single_gate_count = sum(operation.num_subsystems == 1 for operation, _ in sequence)
print(f"Primitive instructions: {len(sequence)}")
print(f"Qubit–qutrit interactions: {interaction_count} (all CClock)")
print(f"Single-qubit or single-qutrit gates: {single_gate_count}")
