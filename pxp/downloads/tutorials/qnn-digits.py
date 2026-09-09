"""Recognize handwritten digits with a quantum neural network

Train a data-reuploading circuit to distinguish handwritten 3s and 6s while evaluating a whole parameter batch with one sweep.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_digits

import fatqat as fq
import fatqat.operations as ops

NUM_QUBITS = 4
NUM_ROUNDS = 4  # 4 rounds x 4 qubits consume the 16 pooled pixels
NUM_PARAMS = NUM_QUBITS * NUM_ROUNDS

digits = load_digits()
subset = (digits.target == 3) | (digits.target == 6)
images = digits.images[subset]
labels = (digits.target[subset] == 6).astype(int)  # 0 for "3", 1 for "6"

rng = np.random.default_rng(0)
order = rng.permutation(len(labels))
images, labels = images[order], labels[order]

pooled = images.reshape(-1, 4, 2, 4, 2).mean(axis=(2, 4))  # 8x8 -> 4x4
features = pooled.reshape(-1, 16) / 16.0 * np.pi

N_TRAIN = 120
X_train, y_train = features[:N_TRAIN], labels[:N_TRAIN]
X_test, y_test = features[N_TRAIN:], labels[N_TRAIN:]
print(f"train {len(y_train)} samples, test {len(y_test)} samples")

# %%
fig, axes = plt.subplots(2, 4, figsize=(8, 4.5))
for ax, image, label in zip(axes.ravel(), pooled[:8], labels[:8]):
    ax.imshow(image, cmap="gray_r", vmin=0, vmax=16)
    ax.set_title(f"true: {'6' if label else '3'}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Model input: 4x4 average-pooled digits")
fig.tight_layout(h_pad=2.5)

# %%
FEATURES = fq.ParameterVector("features", NUM_PARAMS)
WEIGHTS = fq.ParameterVector("weights", NUM_PARAMS)


def build_template():
    """The data-re-uploading circuit, built once with placeholders."""
    program = fq.Program(NUM_QUBITS)
    for r in range(NUM_ROUNDS):
        for q in range(NUM_QUBITS):
            program.add(ops.RY(FEATURES[r * NUM_QUBITS + q]), q)
            program.add(ops.RZ(WEIGHTS[r * NUM_QUBITS + q]), q)
        for q in range(NUM_QUBITS):
            program.add(ops.CX, (q, (q + 1) % NUM_QUBITS))
    return program


template = build_template()

# %%
figure = template.draw("matplotlib")
figure.set_size_inches(16, 4)

# %%
LOGIT_OBSERVABLES = [
    fq.Observable.from_sparse(
        [("Z", (0,), 1.0), ("Z", (1,), 1.0)], num_qubits=NUM_QUBITS
    ),
    fq.Observable.from_sparse(
        [("Z", (2,), 1.0), ("Z", (3,), 1.0)], num_qubits=NUM_QUBITS
    ),
]
estimator = fq.Estimator(fq.simulator.Simulator(method="SV"))


def batch_logits(params, X):
    """Map a batch of samples to class logits with one sweep call."""
    bound = template.assign_parameters({WEIGHTS: params})
    results = estimator.run_sweep(
        bound,
        LOGIT_OBSERVABLES,
        {FEATURES: X},
        shots=0,
    ).result()
    return np.array([result.get_expectation() for result in results])  # (N, 2)

# %%
x0 = rng.uniform(-0.1, 0.1, NUM_PARAMS)
print("logits of the first three test images at initialization:")
print(batch_logits(x0, X_test[:3]))

# %%
def batch_loss(params, X, y, trace=None):
    """Mean softmax cross-entropy over a batch."""
    logits = batch_logits(params, X)
    shifted = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(shifted)
    p /= p.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y]).mean()
    if trace is not None:
        trace.append(loss)
        if len(trace) % 25 == 0:
            print(f"eval {len(trace):4d}  train loss {loss:.4f}")
    return loss


trace = []
result = minimize(
    batch_loss,
    x0,
    args=(X_train, y_train, trace),
    method="COBYLA",
    options={"maxiter": 200, "rhobeg": 0.5},
)
print(f"final train loss {result.fun:.4f} after {len(trace)} evaluations")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trace)
ax.set_xlabel("loss evaluation")
ax.set_ylabel("mean cross-entropy")
ax.set_title("COBYLA training trace")
fig.tight_layout()

# %%
test_logits = batch_logits(result.x, X_test)
test_accuracy = (test_logits.argmax(axis=1) == y_test).mean()
print(f"test accuracy {test_accuracy:.1%} on {len(y_test)} images")

# %%
fig, axes = plt.subplots(3, 4, figsize=(8, 6.5))
offset = N_TRAIN  # pooled/labels indices corresponding to X_test
for k, ax in enumerate(axes.ravel()):
    prediction = test_logits[k].argmax()
    correct = prediction == y_test[k]
    ax.imshow(pooled[offset + k], cmap="gray_r", vmin=0, vmax=16)
    ax.set_title(
        f"pred: {'6' if prediction else '3'}  (true: {'6' if y_test[k] else '3'})",
        color="tab:green" if correct else "tab:red",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Test predictions after training")
fig.tight_layout(h_pad=2.5)
