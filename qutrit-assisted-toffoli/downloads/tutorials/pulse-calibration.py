"""Calibrating a quantum gate

Build a control pulse, collect amplitude-calibration data, and verify an X rotation on several input states.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

np.set_printoptions(precision=6, suppress=True)
final_state_only = {"counts": False, "final_state": True}

reference_program = fq.Program(1)
reference_program.add(ops.RX(np.pi), 0)
reference = (
    fq.simulator.Simulator(method="unitary")
    .run(reference_program, result_config=final_state_only)
    .result()
    .get_unitary()
)
assert reference.shape == (2, 2)
assert np.allclose(reference.conj().T @ reference, np.eye(2), atol=1e-12)
print("Target RX(pi):")
print(reference)

# %%
model_document = fq.emulator.load_model_document("transmon.single")
model = fq.emulator.TransmonModel.from_document(model_document)
backend = fq.emulator.TransmonEmulator(model, method="statevector")

# %%
duration = 12.0  # ns
times = np.linspace(0.0, duration, 81)
envelope = (2 * np.pi / duration) * np.sin(np.pi * times / duration) ** 2


def pulse_program(drive_samples, *, measured=False):
    waveform = fq.emulator.SampledWaveform(tuple(times), tuple(drive_samples))
    control = fq.emulator.PulseControl(model.control.drive("q0"), waveform)
    program = fq.Program(1, 1) if measured else fq.Program(1)
    program.add(ops.PulseOperation(duration, (control,)))
    if measured:
        program.measure(0, 0)
    return program


def make_pulse_program(amplitude_scale):
    return pulse_program(amplitude_scale * envelope)


def run_state(program):
    state = (
        backend.run(program, result_config=final_state_only).result().get_statevector()
    )
    assert state.shape == (3,)
    assert np.all(np.isfinite(state))
    assert np.isclose(np.vdot(state, state).real, 1.0, atol=2e-6, rtol=0)
    return state


initial_scale = 0.8
initial_program = make_pulse_program(initial_scale)
initial_state = run_state(initial_program)
initial_populations = np.abs(initial_state) ** 2
print(f"Duration: {duration:g} ns; samples: {len(times)}")
print(f"Initial peak Rabi rate: {initial_scale * envelope.max():.6f} rad/ns")
print("Initial [P0, P1, P2]:", initial_populations)

# %%
amplitude_scales = np.linspace(0.7, 1.3, 25)
scan_populations = []
for amplitude_scale in amplitude_scales:
    state = run_state(make_pulse_program(amplitude_scale))
    scan_populations.append(np.abs(state) ** 2)
scan_populations = np.array(scan_populations)

best_index = int(np.argmax(scan_populations[:, 1]))
selected_scale = float(amplitude_scales[best_index])
selected_program = make_pulse_program(selected_scale)
selected_state = run_state(selected_program)
selected_populations = np.abs(selected_state) ** 2

assert 0 < best_index < len(amplitude_scales) - 1
assert np.allclose(
    selected_populations, scan_populations[best_index], atol=2e-6, rtol=0
)
assert selected_populations[1] > initial_populations[1] + 0.05
calibrated_parameters = {"duration_ns": duration, "amplitude_scale": selected_scale}

print("Best tested parameters for transfer from |0>:", calibrated_parameters)
for label, scale, populations in (
    ("Initial", initial_scale, initial_populations),
    ("Selected", selected_scale, selected_populations),
):
    print(
        f"{label:8s} a={scale:.3f}: P0={populations[0]:.6f}, "
        f"P1={populations[1]:.6f}, P2={populations[2]:.6f}"
    )

figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for axis, level, factor, label in (
    (axes[0], 1, 100, "Desired-level population P1 (%)"),
    (axes[1], 2, 100, "Leakage P2 (%)"),
):
    axis.plot(amplitude_scales, factor * scan_populations[:, level], ".-", color="C0")
    axis.scatter(
        initial_scale,
        factor * initial_populations[level],
        marker="x",
        s=85,
        color="C1",
        label="Initial",
        zorder=3,
    )
    axis.scatter(
        selected_scale,
        factor * selected_populations[level],
        marker="*",
        s=130,
        color="black",
        label="Selected",
        zorder=3,
    )
    axis.set(xlabel="Amplitude scale a", ylabel=label)
    axis.grid(alpha=0.2)
axes[0].set_title("Amplitude calibration from |0⟩")
axes[1].set_title("Population outside the qubit space")
axes[1].set_ylim(bottom=0)
axes[0].legend()
figure.tight_layout()
plt.show()

# %%
unitary_backend = fq.emulator.TransmonEmulator(model, method="unitary")


def run_unitary(program):
    operator = (
        unitary_backend.run(program, result_config=final_state_only)
        .result()
        .get_unitary()
    )
    assert operator.shape == (3, 3)
    assert np.all(np.isfinite(operator))
    assert np.allclose(operator.conj().T @ operator, np.eye(3), atol=2e-5, rtol=0)
    return operator


probe_labels = ("|0⟩", "|1⟩", "|+⟩", "|+i⟩")
probes = np.array([[1, 0], [0, 1], [1, 1], [1, 1j]], dtype=complex)
probes[2:] /= np.sqrt(2)


def probe_diagnostics(operator):
    overlaps, leakages = [], []
    for psi in probes:
        output = operator @ np.append(psi, 0)
        target = np.append(reference @ psi, 0)
        overlaps.append(abs(np.vdot(target, output)) ** 2)
        leakages.append(abs(output[2]) ** 2)
    return np.array(overlaps), np.array(leakages)


def coherent_x_overlap_and_leakage(operator):
    """Compare the logical operation with RX(pi), retaining leakage loss."""
    logical = operator[:2, :2]
    overlap = abs(np.trace(reference.conj().T @ logical)) ** 2 / 4
    mean_leakage = 1 - np.trace(logical.conj().T @ logical).real / 2
    return float(overlap), float(mean_leakage)


initial_operator = run_unitary(initial_program)
selected_operator = run_unitary(selected_program)
for operator, state in (
    (initial_operator, initial_state),
    (selected_operator, selected_state),
):
    assert np.allclose(operator[:, 0], state, atol=2e-6, rtol=0)

# %%
phase_program = pulse_program(1j * selected_scale * envelope)
phase_operator = run_unitary(phase_program)
operators = (initial_operator, selected_operator, phase_operator)
pulse_labels = ("Initial", "Selected", "Phase-rotated")
probe_results = [probe_diagnostics(operator) for operator in operators]
gate_results = np.array([coherent_x_overlap_and_leakage(u) for u in operators])

all_diagnostics = np.concatenate(
    [gate_results.ravel(), np.array(probe_results).ravel()]
)
assert np.all(np.isfinite(all_diagnostics))
assert np.all((all_diagnostics >= -2e-5) & (all_diagnostics <= 1 + 2e-5))
assert gate_results[1, 0] > gate_results[0, 0]
assert np.allclose(
    np.abs(phase_operator[:, 0]) ** 2, selected_populations, atol=2e-6, rtol=0
)
assert probe_results[1][0][2] > 0.9 and probe_results[2][0][2] < 0.01
assert gate_results[2, 0] < 0.01

print("Pulse          Coherent X overlap    Mean input leakage")
for label, (overlap, leakage) in zip(pulse_labels, gate_results):
    print(f"{label:14s} {overlap:.6f}              {leakage:.6f}")

figure, axes = plt.subplots(1, 2, figsize=(8, 3.8))
positions = np.arange(len(probes))
width = 0.25
for index, (label, (overlaps, leakages)) in enumerate(zip(pulse_labels, probe_results)):
    offset = (index - 1) * width
    hatch = "//" if index == 2 else None
    axes[0].bar(positions + offset, overlaps, width, label=label, hatch=hatch)
    axes[1].bar(positions + offset, 100 * leakages, width, hatch=hatch)
for axis in axes:
    axis.set(xticks=positions, xticklabels=probe_labels, xlabel="Input state")
axes[0].set(
    ylabel="Input-state overlap with RX(π)",
    ylim=(0, 1.08),
    title="Check more than population transfer",
)
figure.legend(
    *axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=9
)
axes[1].set(ylabel="Output leakage (%)", title="Keep the third level visible")
figure.tight_layout(rect=(0, 0, 1, 0.9))
plt.show()

# %%
shots = 1024
measured_program = pulse_program(selected_scale * envelope, measured=True)
counts = (
    backend.run(
        measured_program,
        shots=shots,
        simulation_config={"seed": 7},
        result_config={"counts": True, "final_state": False},
    )
    .result()
    .get_counts()
)
frequency_one = counts.get("1", 0) / shots
expected_one = selected_populations[1] + selected_populations[2]
sampling_tolerance = 5 * np.sqrt(expected_one * (1 - expected_one) / shots) + 1 / shots
assert sum(counts.values()) == shots
assert abs(frequency_one - expected_one) < sampling_tolerance
print("Seeded counts:", counts)
print(f"Measured frequency of 1: {frequency_one:.6f}")
print(f"Expected P1 + P2:        {expected_one:.6f}")
print(f"P1 alone:               {selected_populations[1]:.6f}")

# %%
alpha = 2 * np.pi * model_document["parameters"]["subsystems"]["q0"]["anharmonicity"]
envelope_derivative = (
    selected_scale
    * (2 * np.pi / duration)
    * (np.pi / duration)
    * np.sin(2 * np.pi * times / duration)
)


def corrected_samples(beta):
    return selected_scale * envelope - 1j * beta * envelope_derivative / alpha


betas = np.linspace(0.0, 1.0, 11)
shape_results = []
for beta in betas:
    operator = run_unitary(pulse_program(corrected_samples(beta)))
    shape_results.append(coherent_x_overlap_and_leakage(operator))
shape_results = np.array(shape_results)
shape_index = int(np.argmax(shape_results[:, 0]))
selected_beta = float(betas[shape_index])
corrected_program = pulse_program(corrected_samples(selected_beta))
corrected_operator = run_unitary(corrected_program)
corrected_metrics = coherent_x_overlap_and_leakage(corrected_operator)
corrected_state = run_state(corrected_program)
corrected_overlaps, corrected_leakages = probe_diagnostics(corrected_operator)

assert np.all(np.isfinite(shape_results))
assert np.all((shape_results >= -2e-5) & (shape_results <= 1 + 2e-5))
assert np.allclose(shape_results[0], gate_results[1], atol=2e-6, rtol=0)
assert np.allclose(corrected_metrics, shape_results[shape_index], atol=2e-6, rtol=0)
assert np.allclose(corrected_operator[:, 0], corrected_state, atol=2e-6, rtol=0)
assert 1 - corrected_metrics[0] < (1 - gate_results[1, 0]) / 10
assert corrected_metrics[1] < gate_results[1, 1]
assert np.all(corrected_overlaps > 0.99)

print(f"Fixed amplitude: {selected_scale:.3f}; best tested beta: {selected_beta:.2f}")
print(
    f"Coherent X overlap: {corrected_metrics[0]:.6f}; "
    f"mean input leakage: {corrected_metrics[1]:.6f}"
)
print("Input    Overlap    Leakage")
for label, overlap, leakage in zip(
    probe_labels, corrected_overlaps, corrected_leakages
):
    print(f"{label:6s}   {overlap:.6f}   {leakage:.6f}")

figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for axis, values, label in (
    (axes[0], 1 - shape_results[:, 0], "Coherent X error (1 − overlap)"),
    (axes[1], shape_results[:, 1], "Mean input leakage"),
):
    axis.semilogy(betas, values, ".-", color="C0")
    axis.scatter(
        selected_beta,
        values[shape_index],
        marker="*",
        s=130,
        color="black",
        label="Selected by X overlap",
        zorder=3,
    )
    axis.set(xlabel="Shape coefficient β", ylabel=label)
    axis.grid(alpha=0.2)
axes[0].set_title("Shape scan at fixed amplitude")
axes[1].set_title("Leakage alone gives a different choice")
axes[0].legend(fontsize=8)
figure.tight_layout()
plt.show()
