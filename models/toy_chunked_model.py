"""
Toy action-chunking model for controlled Gibbs overshoot experiments.

A small MLP with sinusoidal positional encodings that outputs a CHUNK
of future actions given an instruction embedding.  The sinusoidal basis
acts like a truncated Fourier series: when the model must reconstruct a
step function at an instruction boundary, the finite basis produces
Gibbs ringing naturally.

No GPU or deep-learning framework required -- pure NumPy.

Architecture
------------
  instruction text  -->  hash  -->  embedding (embed_dim)
  timestep index    -->  sinusoidal positional encoding (embed_dim)
          |                                    |
          +-------- concatenate ---------------+
                         |
                     MLP (2 hidden layers, tanh activation)
                         |
                     action (action_dim)

The model predicts one action per timestep, but the chunk of
``chunk_size`` actions is generated in a single call (vectorized).

Training target: for each instruction, the model should output a
constant action vector at every timestep.  The positional encoding
gives the model temporal structure, and the truncated sinusoidal basis
ensures Gibbs-like ringing when the target is a step function.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (truncated Fourier basis)
# ---------------------------------------------------------------------------

def sinusoidal_encoding(positions: np.ndarray, dim: int) -> np.ndarray:
    """
    Sinusoidal positional encoding a la Vaswani et al.

    Parameters
    ----------
    positions : (T,) array of integer timestep indices
    dim : embedding dimension (must be even)

    Returns
    -------
    (T, dim) array of positional encodings
    """
    assert dim % 2 == 0, "dim must be even"
    positions = np.asarray(positions, dtype=float)[:, None]  # (T, 1)
    freq_indices = np.arange(dim // 2, dtype=float)[None, :]  # (1, D/2)
    # Frequencies decrease geometrically: 1, 1/10000^(2/d), ...
    # Using a SMALL max_period so that the basis is truly truncated
    # and higher harmonics are present in the encoding.
    omega = 1.0 / (100.0 ** (2.0 * freq_indices / dim))
    angles = positions * omega  # (T, D/2)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)  # (T, D)


# ---------------------------------------------------------------------------
# Simple MLP in NumPy
# ---------------------------------------------------------------------------

def _he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He (Kaiming) initialization for tanh networks."""
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, size=(fan_in, fan_out))


@dataclass
class MLPParams:
    """Learnable parameters for a 2-hidden-layer MLP."""
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray


def _init_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    rng: np.random.Generator,
) -> MLPParams:
    return MLPParams(
        W1=_he_init(input_dim, hidden_dim, rng),
        b1=np.zeros(hidden_dim),
        W2=_he_init(hidden_dim, hidden_dim, rng),
        b2=np.zeros(hidden_dim),
        W3=_he_init(hidden_dim, output_dim, rng),
        b3=np.zeros(output_dim),
    )


def _forward_mlp(params: MLPParams, x: np.ndarray) -> np.ndarray:
    """Forward pass: tanh-tanh-linear."""
    h = np.tanh(x @ params.W1 + params.b1)
    h = np.tanh(h @ params.W2 + params.b2)
    return h @ params.W3 + params.b3


# ---------------------------------------------------------------------------
# Toy Chunked Action Model
# ---------------------------------------------------------------------------

@dataclass
class ToyChunkedModel:
    """
    Tiny action-chunking model (pure NumPy, no autograd).

    Attributes
    ----------
    chunk_size : int
        Number of future timesteps predicted per call.
    action_dim : int
        Dimensionality of the action space.
    embed_dim : int
        Instruction and positional embedding dimension.
    hidden_dim : int
        MLP hidden layer width.
    """

    chunk_size: int = 50
    action_dim: int = 7
    embed_dim: int = 32
    hidden_dim: int = 64
    seed: int = 42

    # Learnable instruction embeddings: instruction_text -> embedding
    _instruction_embeds: dict[str, np.ndarray] = field(
        default_factory=dict, repr=False
    )
    _mlp: MLPParams | None = field(default=None, repr=False)
    _rng: np.random.Generator = field(default=None, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._mlp = _init_mlp(
            input_dim=2 * self.embed_dim,  # instruction + positional
            hidden_dim=self.hidden_dim,
            output_dim=self.action_dim,
            rng=self._rng,
        )

    # ----- instruction embedding -----

    def _get_instruction_embed(self, instruction: str) -> np.ndarray:
        """
        Get or create a learnable embedding for an instruction.

        Uses a hash-seeded random vector as initialization.
        During training this will be updated via gradient descent.
        """
        if instruction not in self._instruction_embeds:
            h = int(hashlib.sha256(instruction.encode()).hexdigest(), 16)
            inst_rng = np.random.default_rng(h % (2**63))
            self._instruction_embeds[instruction] = inst_rng.normal(
                0, 0.5, size=self.embed_dim
            )
        return self._instruction_embeds[instruction]

    # ----- forward pass -----

    def predict_chunk(self, instruction: str) -> np.ndarray:
        """
        Predict a chunk of actions for the given instruction.

        Returns
        -------
        actions : (chunk_size, action_dim) array
        """
        # Instruction embedding: (embed_dim,) -> broadcast to (T, embed_dim)
        inst_embed = self._get_instruction_embed(instruction)
        inst_tiled = np.tile(inst_embed, (self.chunk_size, 1))  # (T, E)

        # Positional encoding: (T, embed_dim)
        positions = np.arange(self.chunk_size)
        pos_embed = sinusoidal_encoding(positions, self.embed_dim)  # (T, E)

        # Concatenate and forward through MLP
        x = np.concatenate([inst_tiled, pos_embed], axis=-1)  # (T, 2E)
        actions = _forward_mlp(self._mlp, x)  # (T, action_dim)
        return actions

    # ----- training -----

    def train(
        self,
        task_targets: dict[str, np.ndarray],
        n_epochs: int = 2000,
        lr: float = 0.005,
        verbose: bool = True,
    ) -> list[float]:
        """
        Train the model to produce constant action trajectories.

        Parameters
        ----------
        task_targets : dict
            Maps instruction string to target action vector (action_dim,).
            The model should output this constant vector at every timestep
            in the chunk.
        n_epochs : int
            Number of SGD epochs.
        lr : float
            Learning rate.

        Returns
        -------
        losses : list of float
            MSE loss per epoch.
        """
        losses = []
        instructions = list(task_targets.keys())

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            for instruction in instructions:
                target = np.asarray(task_targets[instruction], dtype=float)
                # Target is constant across all timesteps
                target_chunk = np.tile(target, (self.chunk_size, 1))  # (T, D)

                # Forward pass
                pred = self.predict_chunk(instruction)  # (T, D)
                residual = pred - target_chunk  # (T, D)
                loss = 0.5 * np.mean(residual ** 2)
                epoch_loss += loss

                # ----- Manual backprop (simple finite-difference gradient) -----
                # We use analytical gradients for the linear layers and
                # numerical gradients for instruction embeddings.
                self._backprop_step(instruction, target_chunk, lr)

            epoch_loss /= len(instructions)
            losses.append(epoch_loss)

            if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
                print(f"  Epoch {epoch:5d}/{n_epochs}  loss={epoch_loss:.6f}")

        return losses

    def _backprop_step(
        self,
        instruction: str,
        target: np.ndarray,
        lr: float,
    ):
        """
        One gradient-descent step via manual backprop through the MLP.

        The network is: x -> [W1,b1,tanh] -> [W2,b2,tanh] -> [W3,b3,linear] -> y
        """
        p = self._mlp
        T = self.chunk_size

        # --- Forward pass (save intermediates) ---
        inst_embed = self._get_instruction_embed(instruction)
        inst_tiled = np.tile(inst_embed, (T, 1))
        positions = np.arange(T)
        pos_embed = sinusoidal_encoding(positions, self.embed_dim)
        x = np.concatenate([inst_tiled, pos_embed], axis=-1)  # (T, 2E)

        z1 = x @ p.W1 + p.b1          # (T, H)
        h1 = np.tanh(z1)              # (T, H)
        z2 = h1 @ p.W2 + p.b2         # (T, H)
        h2 = np.tanh(z2)              # (T, H)
        out = h2 @ p.W3 + p.b3        # (T, D)

        # --- Loss gradient: d(0.5*mean(residual^2))/d(out) ---
        residual = out - target        # (T, D)
        d_out = residual / T           # (T, D)

        # --- Layer 3 (linear) ---
        dW3 = h2.T @ d_out            # (H, D)
        db3 = np.sum(d_out, axis=0)   # (D,)
        d_h2 = d_out @ p.W3.T         # (T, H)

        # --- Layer 2 (tanh) ---
        d_z2 = d_h2 * (1 - h2 ** 2)   # (T, H)
        dW2 = h1.T @ d_z2             # (H, H)
        db2 = np.sum(d_z2, axis=0)    # (H,)
        d_h1 = d_z2 @ p.W2.T          # (T, H)

        # --- Layer 1 (tanh) ---
        d_z1 = d_h1 * (1 - h1 ** 2)   # (T, H)
        dW1 = x.T @ d_z1              # (2E, H)
        db1 = np.sum(d_z1, axis=0)    # (H,)
        d_x = d_z1 @ p.W1.T           # (T, 2E)

        # --- Update MLP weights ---
        p.W3 -= lr * dW3
        p.b3 -= lr * db3
        p.W2 -= lr * dW2
        p.b2 -= lr * db2
        p.W1 -= lr * dW1
        p.b1 -= lr * db1

        # --- Update instruction embedding ---
        # d_x[:, :embed_dim] is the gradient w.r.t. the tiled instruction embed
        d_inst = np.sum(d_x[:, : self.embed_dim], axis=0)  # (E,)
        self._instruction_embeds[instruction] -= lr * d_inst


# ---------------------------------------------------------------------------
# PolicyAdapter interface (compatible with InstructionSwapRunner)
# ---------------------------------------------------------------------------

class ToyChunkedPolicy:
    """
    Wraps ToyChunkedModel to satisfy the PolicyAdapter protocol
    used by InstructionSwapRunner.

    Like ChunkedVLAPolicy, this maintains an internal action buffer.
    Actions from the current chunk are returned one at a time.
    When the buffer is exhausted, a new chunk is generated.
    When the instruction changes mid-chunk, the stale buffer is NOT
    regenerated -- this is what creates the temporal coupling for
    Gibbs overshoot.
    """

    def __init__(self, model: ToyChunkedModel):
        self.model = model
        self._action_buffer: np.ndarray | None = None
        self._buffer_idx: int = 0
        self._last_instruction: str | None = None

    @property
    def action_dim(self) -> int:
        return self.model.action_dim

    @property
    def chunk_size(self) -> int:
        return self.model.chunk_size

    def reset(self):
        self._action_buffer = None
        self._buffer_idx = 0
        self._last_instruction = None

    def predict(self, observation: np.ndarray, instruction: str) -> np.ndarray:
        """
        Return one action from the current chunk buffer.

        Key behavior: mid-chunk instruction changes do NOT trigger
        re-planning. The stale actions from the old instruction
        persist until the buffer is exhausted.
        """
        need_new_chunk = (
            self._action_buffer is None
            or self._buffer_idx >= self._action_buffer.shape[0]
        )

        if need_new_chunk:
            self._action_buffer = self.model.predict_chunk(instruction)
            self._buffer_idx = 0
            self._last_instruction = instruction

        action = self._action_buffer[self._buffer_idx]
        self._buffer_idx += 1
        return np.asarray(action, dtype=np.float64)
