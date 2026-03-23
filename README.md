# Shape Counter
<img width="2558" height="1773" alt="image" src="https://github.com/user-attachments/assets/3a5fe440-f528-4f21-9595-1578bfca98dd" />

A CNN that counts circles and squares in 64×64 black-and-white images. All shapes are **outlines only** — 1px Bresenham-rasterised borders, no fills. Squares can be rotated to any angle. The network outputs two numbers: how many circles and how many squares are in the image.

Trained entirely from synthetic data generated on-the-fly (no dataset files needed), with a browser-based playground for interactive inference.

## The Problem

Given a 64×64 binary image containing a mix of:

- **Circle outlines** — varying radius (4–18 px), placed anywhere (including partially off-canvas)
- **Rotated square outlines** — varying size (4–16 px half-side), any rotation angle, also partially clippable

The model must output `(num_circles, num_squares)` as a regression task. Shapes overlap freely — where two outlines cross, the intersection is just a few scattered pixels, not a separate shape.

Key difficulties:

- **Counting, not classification** — the output is two continuous values, not a class label
- **Overlapping outlines** — shapes can cross each other arbitrarily
- **Partial visibility** — shapes may be clipped by the canvas border, leaving only a fragment visible
- **Rotation invariance for squares** — a square at 5° tilt looks almost axis-aligned but is still one square, not two
- **Scale variation** — small and large shapes coexist in the same image
- **High density** — up to 15 circles + 15 squares on a 64×64 canvas

## Training

Everything lives in a single `src/main.rs`. No dataset files, no preprocessing — training data is generated procedurally at each batch using Bresenham circle and line rasterisers.

**Data generation features:**

- Shape counts drawn from a mixture distribution (circles-only, squares-only, empty canvas, mixed, heavy, lopsided scenarios)
- Three placement strategies per shape: normal (inside canvas), edge (partially clipped), extreme (mostly off-canvas)
- Square rotation angles biased toward near-axis-aligned tilts (the hardest case)
- Intensity noise augmentation during training (pixels multiplied by random values in [0.7, 1.0])

**Training setup:**

- Loss: L1 (mean absolute error) — better than MSE for counting tasks
- Optimizer: Adam with cosine annealing LR schedule (1e-3 → 1e-5)
- Best-only checkpointing: saves a single `shape_counter_best.json` that is overwritten only when validation loss improves

### Prerequisites

- Rust (stable, 2021 edition)
- NVIDIA GPU with CUDA support
- CUDA toolkit installed

### Build & Run

```bash
cd shape-counter
cargo run --release
```

## Playground

`playground.html` is a self-contained single-file browser app for interactive inference. No server, no build step — just open it in a browser.

### How to use

1. Open `playground.html` in a browser (works from `file://`)
2. Click **Load Model JSON** and select your `shape_counter_best.json` checkpoint
3. Draw shapes on the 64×64 canvas — predictions update automatically

### Drawing controls

| Tool             | Action                                                                                   |
| ---------------- | ---------------------------------------------------------------------------------------- |
| **Circle** (`C`) | Click and drag to set centre + radius, release to place                                  |
| **Square** (`S`) | Click and drag to set centre + size, release, then move mouse to rotate, click to commit |
| **Eraser** (`E`) | Click and drag to erase pixels                                                           |
| **Clear**        | Wipe the canvas                                                                          |
| `Space`          | Manually trigger inference                                                               |
| `Esc`            | Cancel current shape                                                                     |

The square tool has a **two-phase interaction**: drag sets the size, then mouse movement controls rotation angle, and a second click commits. This lets you place squares at any rotation.

### How it works

The playground implements the **full CNN forward pass in plain JavaScript** — no WASM, no server, no dependencies. It parses the burn JSON checkpoint, decodes the raw bytes into Float32Arrays, and runs conv2d, batch norm, ReLU, max pool, global average pool, and linear layers as nested loops. Inference takes ~50–200ms depending on your CPU.

The model JSON is loaded via the browser's `FileReader` API from a native file picker dialog. The file never leaves your machine.

### Test buttons

Two built-in test image generators let you verify the model is working:

- **🧪 Test: 3 circles, 2 squares** — a light test image
- **🧪 Heavy: 10 circles, 8 squares** — a dense image closer to the training distribution

## Project Structure

```
shape-counter/
├── Cargo.toml          # burn 0.20 + cuda + autodiff + fusion
├── src/
│   └── main.rs         # model, data generation, training loop — everything
├── playground.html     # browser inference playground (standalone)
└── checkpoints/        # created at runtime
    └── model_snaphost_N.json
```

## Notes

- The burn CUDA backend (`burn-cuda`) uses CubeCL to JIT-compile PTX kernels at runtime — no libtorch dependency, pure Rust all the way down
- Burn stores `Linear` weights as `[d_input, d_output]` and computes `y = x @ W + b` (not transposed like PyTorch) — the playground's JS matches this layout
- Batch norm uses inference-mode statistics (`running_mean`, `running_var`) in the playground — dropout is skipped (no-op at inference)
- The checkpoint JSON encodes tensors as `{ bytes: [u8...], shape: [dims...] }` wrapped in `{ id: "...", param: { bytes, shape } }` — the playground's parser handles this structure
