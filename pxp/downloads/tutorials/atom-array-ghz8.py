"""Entangle eight atoms into a GHZ state

Use dynamic Pair and Unpair operations to build an eight-atom GHZ state, then test both its correlations and coherent phase.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

np.set_printoptions(precision=3, suppress=True)

NUM_ATOMS = 8

# %%
def native_h(program: fq.Program, target: int) -> None:
    """Hadamard in the native gate set: ``RZ(pi)`` then ``RY(pi/2)``."""
    program.add(ops.RZ(np.pi), target)
    program.add(ops.RY(np.pi / 2), target)


def native_cx(program: fq.Program, control: int, target: int) -> None:
    """``CX(control -> target)`` as ``H(target) CZ H(target)``.

    The ``CZ`` in the middle is valid only while ``control`` and ``target`` are
    currently paired; otherwise the backend raises
    :class:`~fatqat.errors.BackendValidationError`.
    """
    native_h(program, target)
    program.add(ops.CZ, (control, target))
    native_h(program, target)

# %%
CX_LAYERS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 4),),
    ((0, 2), (4, 6)),
    ((0, 1), (2, 3), (4, 5), (6, 7)),
)

for layer_index, layer in enumerate(CX_LAYERS, start=1):
    print(f"layer {layer_index}: " + " || ".join(f"CX{pair}" for pair in layer))

# %%
def build_ghz8_program(*, measure: bool = True) -> fq.Program:
    """Assemble the eight-atom GHZ program.

    With ``measure=True`` every atom is read into a classical bit at the end,
    which is what the counts experiment needs. With ``measure=False`` the
    program has no classical register, leaving the coherent final state for the
    :class:`~fatqat.Estimator` to interrogate.
    """
    program = fq.Program(NUM_ATOMS, NUM_ATOMS if measure else 0)

    # Sites start empty; load one |0> atom into each of the eight traps.
    program.add(ops.Put, tuple(range(NUM_ATOMS)))

    # Seed the tree: put atom 0 into |+>, the root the branches grow from.
    native_h(program, 0)

    for layer in CX_LAYERS:
        for pair in layer:  # transport the layer's atoms together
            program.add(ops.Pair, pair)
        for control, target in layer:  # one parallel layer of CX = H CZ H
            native_cx(program, control, target)
        program.add(ops.Barrier, tuple(range(NUM_ATOMS)))  # visual layer marker
        for pair in layer:  # move the pairs apart again
            program.add(ops.Unpair, pair)

    if measure:
        program.measure_all()
    return program


ghz_program = build_ghz8_program()

# %%
shots = 2_000
backend = fq.simulator.AtomArraySimulator()
counts = (
    backend.run(
        ghz_program,
        shots=shots,
        simulation_config={"seed": 7},
    )
    .result()
    .get_counts()
)

print("Counts:", counts)

# %%
all_zero, all_one = "0" * NUM_ATOMS, "1" * NUM_ATOMS
observed = np.array([counts.get(all_zero, 0), counts.get(all_one, 0)]) / shots

figure, axis = plt.subplots(figsize=(7, 4))
positions = np.arange(2)
axis.bar(positions, observed, width=0.55, label="seeded simulation")
axis.scatter(positions, [0.5, 0.5], color="black", marker="_", s=350, label="ideal")
axis.set(
    xticks=positions,
    xticklabels=(all_zero, all_one),
    xlabel="Measured bitstring",
    ylabel="Frequency",
    ylim=(0, 0.6),
    title="GHZ$_8$ measurement frequencies",
)
axis.legend()
figure.tight_layout()
plt.show()

# %%
zz_observables = []
for i in range(NUM_ATOMS - 1):
    label = ["I"] * NUM_ATOMS
    label[i] = label[i + 1] = "Z"
    zz_observables.append(fq.Observable([("".join(label), 1.0)]))
x_parity = fq.Observable([("X" * NUM_ATOMS, 1.0)])

estimator = fq.Estimator(fq.simulator.AtomArraySimulator())
values = (
    estimator.run(
        build_ghz8_program(measure=False),
        zz_observables + [x_parity],
    )
    .result()
    .get_expectation()
)

for i, value in enumerate(values[:-1]):
    print(f"<Z{i}Z{i + 1}> = {value:+.6f}")
print(f"<{'X' * NUM_ATOMS}> = {values[-1]:+.6f}")

# %%
noise = fq.NoiseModel()
noise.add(fq.noise.Loss(p=0.01), operation=ops.Pair)
noise.add(fq.noise.Loss(p=0.01), operation=ops.Unpair)

lossy_backend = fq.simulator.AtomArraySimulator(noise=noise)
lossy_counts = (
    lossy_backend.run(
        ghz_program,
        shots=shots,
        simulation_config={"seed": 7},
    )
    .result()
    .get_counts()
)

lost_shots = sum(n for bitstring, n in lossy_counts.items() if "2" in bitstring)
print(f"{lost_shots}/{shots} shots lost at least one atom (a '2' in the readout)")
print("Most frequent outcomes under 1% loss per move:")
for bitstring, n in sorted(lossy_counts.items(), key=lambda kv: -kv[1])[:6]:
    print(f"  {bitstring}: {n}")
