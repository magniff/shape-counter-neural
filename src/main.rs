// shape-counter/src/main.rs
//
// A CNN that counts circles and squares in 64×64 binary images.
// All shapes are outlines only — 1px wide borders, no fills.
// Trained with supervised learning — we generate labelled data on-the-fly.
//
// ── Architecture ──────────────────────────────────────────────────────────
//
//   Input: [B, 1, 64, 64]   (batch of single-channel binary images)
//
//   ConvBlock × 4:
//     Conv2d(k=3, pad=1) → BatchNorm → ReLU → MaxPool(2×2, stride=2)
//     Channels: 1→64 → 128 → 128 → 128
//     Spatial:  64→32 → 16 → 8  → 4
//
//   AdaptiveAvgPool2d → [B, 128, 1, 1]    (global average pooling)
//   Flatten           → [B, 256]
//
//   FC head:
//     Linear(256, 128) → ReLU → Dropout(0.3)
//     Linear(128, 64)  → ReLU → Dropout(0.3)
//     Linear(64, 2)    → raw output: [num_circles, num_squares]
//
//   Loss: MSE between predicted counts and ground-truth counts.
//   Optimizer: Adam (lr = 1e-3 with cosine decay).
//
// ── Why this works on real hardware ──────────────────────────────────────
//
//   Total params ≈ 880 K.  At fp32 that's ~3.5 MB of weights — fits entirely
//   in L2 cache on any GPU from Pascal onward.  The bottleneck is the conv
//   layers: cuDNN (or CubeCL's generated kernels in burn-cuda) will pick an
//   implicit-GEMM or Winograd-based algorithm for the 3×3 convolutions.
//   With 64×64 inputs and batch size 64, each conv layer operates on small
//   tiles that stay resident in shared memory / registers, so we're firmly
//   compute-bound, not memory-bound.
//
//   The fusion backend merges consecutive elementwise ops (BN's scale+bias,
//   ReLU) into a single kernel launch, cutting launch overhead roughly in
//   half for each conv block.
//
//   MaxPool2d at stride 2 halves both spatial dims per block, so by the time
//   we reach the FC head the tensor is 256×1×1 — the GEMM for the linear
//   layers is trivially small.
//
// ── Data generation strategy ─────────────────────────────────────────────
//
//   We draw 0–15 circles and 0–15 squares per image, all as outlines only
//   (1px borders).  Circles are rasterised with Bresenham midpoint;
//   squares are 4 Bresenham lines connecting rotated corner vertices.
//
//   Each shape is placed using one of three strategies:
//     • 50% "normal"  — fully or mostly inside the canvas
//     • 30% "edge"    — partially clipped by the border
//     • 20% "extreme" — mostly off-canvas, only a fragment visible
//
//   Shapes that extend beyond [0, 64) are silently clipped at the pixel
//   level (the `set` method discards out-of-bounds writes).  The label
//   counts every shape regardless of visibility — a half-circle at the
//   edge still counts as 1 circle.  This trains the network to recognise
//   partial outlines and extrapolate from visible fragments.
//
//   Overlaps happen naturally — when two outlines cross, the intersection
//   is just a few scattered pixels.  The network learns to trace closed
//   curves and distinguish them by curvature: circles have smooth arcs,
//   squares have straight edges meeting at sharp corners.

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder};
use burn::tensor::backend::AutodiffBackend;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use std::time::Instant;

/// A single conv-block: Conv2d → BatchNorm → ReLU → MaxPool(2×2).
#[derive(Module, Debug)]
struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    activation: Relu,
    pool: MaxPool2d,
}

#[derive(Config, Debug)]
struct ConvBlockConfig {
    in_channels: usize,
    out_channels: usize,
}

impl ConvBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ConvBlock<B> {
        let conv = Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .init(device);

        let bn = BatchNormConfig::new(self.out_channels).init(device);
        let activation = Relu::new();
        let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        ConvBlock {
            conv,
            bn,
            activation,
            pool,
        }
    }
}

impl<B: Backend> ConvBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        let x = self.activation.forward(x);
        self.pool.forward(x)
    }
}

/// The full shape-counting CNN (v2).
///
/// Channel progression: 1 → 64 → 128 → 128 → 128
///   • Wider early layers (64 instead of 32) give more capacity to
///     capture fine texture differences (arc curvature vs straight edges)
///     at high spatial resolution (64×64 and 32×32).
///   • Narrower late layers (128 instead of 256) reduce parameters in
///     the FC head without hurting — by block3/block4 the spatial dims
///     are 8×8 and 4×4, so there's little spatial info left anyway.
///
/// Spatial: 64→32→16→8→4, then GAP to 1×1.
///
/// Total params ≈ 650 K (down from 880 K in v1, but better distributed).
#[derive(Module, Debug)]
struct ShapeCounter<B: Backend> {
    block1: ConvBlock<B>,
    block2: ConvBlock<B>,
    block3: ConvBlock<B>,
    block4: ConvBlock<B>,
    gap: AdaptiveAvgPool2d,
    fc1: Linear<B>,
    drop1: Dropout,
    fc2: Linear<B>,
    drop2: Dropout,
    fc3: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
struct ShapeCounterConfig {}

impl ShapeCounterConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ShapeCounter<B> {
        ShapeCounter {
            block1: ConvBlockConfig::new(1, 64).init(device), // was 1→32
            block2: ConvBlockConfig::new(64, 128).init(device), // was 32→64
            block3: ConvBlockConfig::new(128, 128).init(device), // was 64→128
            block4: ConvBlockConfig::new(128, 128).init(device), // was 128→256
            gap: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc1: LinearConfig::new(128, 128).init(device), // was 256→128
            drop1: DropoutConfig::new(0.3).init(),
            fc2: LinearConfig::new(128, 64).init(device), // same
            drop2: DropoutConfig::new(0.3).init(),
            fc3: LinearConfig::new(64, 2).init(device), // same
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> ShapeCounter<B> {
    fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.block1.forward(images);
        let x = self.block2.forward(x);
        let x = self.block3.forward(x);
        let x = self.block4.forward(x);

        let x = self.gap.forward(x);

        let [batch, channels, _h, _w] = x.dims();
        let x = x.reshape([batch, channels]);

        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);
        let x = self.drop1.forward(x);

        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        let x = self.drop2.forward(x);

        self.fc3.forward(x)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Synthetic data generation
// ═══════════════════════════════════════════════════════════════════════════

const IMG_SIZE: usize = 64;

/// A 64×64 canvas stored as row-major f32 (0.0 = black, 1.0 = white).
struct Canvas {
    pixels: Vec<f32>,
}

impl Canvas {
    fn new() -> Self {
        Canvas {
            pixels: vec![0.0; IMG_SIZE * IMG_SIZE],
        }
    }

    #[inline]
    fn set(&mut self, x: i32, y: i32) {
        if x >= 0 && x < IMG_SIZE as i32 && y >= 0 && y < IMG_SIZE as i32 {
            self.pixels[y as usize * IMG_SIZE + x as usize] = 1.0;
        }
    }

    /// Draw a circle (outline) using the midpoint algorithm.
    /// This is the classic Bresenham circle — no floating point, just
    /// integer adds and comparisons.  Runs in O(radius) iterations.
    fn draw_circle(&mut self, cx: i32, cy: i32, r: i32) {
        let mut x = r;
        let mut y = 0i32;
        let mut err = 1 - r;

        while x >= y {
            // Draw all 8 octants.
            self.set(cx + x, cy + y);
            self.set(cx - x, cy + y);
            self.set(cx + x, cy - y);
            self.set(cx - x, cy - y);
            self.set(cx + y, cy + x);
            self.set(cx - y, cy + x);
            self.set(cx + y, cy - x);
            self.set(cx - y, cy - x);

            y += 1;
            if err < 0 {
                err += 2 * y + 1;
            } else {
                x -= 1;
                err += 2 * (y - x) + 1;
            }
        }
    }

    /// Draw a line from (x0,y0) to (x1,y1) using Bresenham's algorithm.
    /// Runs in O(max(|dx|,|dy|)) with only integer arithmetic — no divides.
    fn draw_line(&mut self, mut x0: i32, mut y0: i32, x1: i32, y1: i32) {
        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        loop {
            self.set(x0, y0);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x0 += sx;
            }
            if e2 < dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    /// Draw a rotated square (outline only — 4 edges as Bresenham lines).
    /// `cx, cy` = centre, `half_side` = half the side length,
    /// `angle` = rotation in radians.
    ///
    /// We compute the 4 corner vertices via a 2×2 rotation matrix,
    /// then draw 4 line segments connecting them.
    fn draw_rotated_square(&mut self, cx: f64, cy: f64, half_side: f64, angle: f64) {
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let corners_local = [
            (-half_side, -half_side),
            (half_side, -half_side),
            (half_side, half_side),
            (-half_side, half_side),
        ];

        let corners: Vec<(i32, i32)> = corners_local
            .iter()
            .map(|&(lx, ly)| {
                let rx = lx * cos_a - ly * sin_a + cx;
                let ry = lx * sin_a + ly * cos_a + cy;
                (rx.round() as i32, ry.round() as i32)
            })
            .collect();

        for i in 0..4 {
            let (x0, y0) = corners[i];
            let (x1, y1) = corners[(i + 1) % 4];
            self.draw_line(x0, y0, x1, y1);
        }
    }
}

/// Outcome of generating one training image.
struct Sample {
    pixels: Vec<f32>, // IMG_SIZE * IMG_SIZE, row-major
    num_circles: f32,
    num_squares: f32,
}

/// Generate a single training sample.
///
/// All shapes are outlines only (1px wide borders).  The drawing primitives
/// (`set`, `draw_circle`, `draw_line`) silently clip to the canvas — any
/// pixel outside [0, IMG_SIZE) is simply discarded.  This means we can
/// freely place shapes that extend beyond the borders.
///
/// We use three placement strategies per shape, chosen randomly:
///   • ~50%  "normal"  — centre is inside the canvas with a small margin,
///                        shape is mostly or fully visible.
///   • ~30%  "edge"    — centre is near or on the border, shape is
///                        partially clipped (20–80% visible).
///   • ~20%  "extreme" — centre can be well outside the canvas, leaving
///                        only a small arc or corner visible.
///
/// The label always counts the shape regardless of how much is visible.
/// This forces the network to recognise partial outlines — a half-circle
/// at the edge is still 1 circle.
fn generate_sample(rng: &mut StdRng) -> Sample {
    let mut canvas = Canvas::new();
    let sz = IMG_SIZE as f64;
    let szi = IMG_SIZE as i32;

    // Shape count selection — we use a mixture of scenarios so the
    // training set includes plenty of corner cases:
    //
    //   30%  mixed        — both circles and squares, 1–15 each
    //   20%  circles-only — 1–15 circles, 0 squares
    //   20%  squares-only — 0 circles, 1–15 squares
    //   10%  empty        — 0 circles, 0 squares (blank canvas)
    //   10%  heavy mixed  — 8–15 of each (stress test, very dense)
    //   10%  lopsided     — many of one kind (5–15), few of other (0–2)
    let scenario: f64 = rng.gen();
    let (num_circles, num_squares) = if scenario < 0.30 {
        // Mixed
        (rng.gen_range(1..=15u32), rng.gen_range(1..=15u32))
    } else if scenario < 0.50 {
        // Circles only
        (rng.gen_range(1..=15u32), 0)
    } else if scenario < 0.70 {
        // Squares only
        (0, rng.gen_range(1..=15u32))
    } else if scenario < 0.80 {
        // Empty canvas
        (0, 0)
    } else if scenario < 0.90 {
        // Heavy mixed (dense images)
        (rng.gen_range(8..=15u32), rng.gen_range(8..=15u32))
    } else {
        // Lopsided: many of one, 0–2 of the other
        if rng.gen_bool(0.5) {
            (rng.gen_range(5..=15u32), rng.gen_range(0..=2u32))
        } else {
            (rng.gen_range(0..=2u32), rng.gen_range(5..=15u32))
        }
    };

    // Draw circle outlines.
    for _ in 0..num_circles {
        let r = rng.gen_range(4..=18) as i32;
        let roll: f64 = rng.gen();
        let (cx, cy) = if roll < 0.5 {
            // Normal: fully or mostly inside.
            let cx = rng.gen_range(r..(szi - r));
            let cy = rng.gen_range(r..(szi - r));
            (cx, cy)
        } else if roll < 0.8 {
            // Edge: centre near border, partially clipped.
            let cx = rng.gen_range(-(r / 2)..(szi + r / 2));
            let cy = rng.gen_range(-(r / 2)..(szi + r / 2));
            (cx, cy)
        } else {
            // Extreme: centre can be well outside, only a small arc visible.
            let cx = rng.gen_range(-r..(szi + r));
            let cy = rng.gen_range(-r..(szi + r));
            (cx, cy)
        };
        canvas.draw_circle(cx, cy, r);
    }

    // Draw rotated square outlines.
    for _ in 0..num_squares {
        let half_side = rng.gen_range(4.0..=16.0_f64);
        let angle = rng.gen_range(0.0..std::f64::consts::PI * 2.0);
        let roll: f64 = rng.gen();
        let (cx, cy) = if roll < 0.5 {
            // Normal: mostly inside.
            let margin = half_side * 0.5;
            let cx = rng.gen_range(margin..(sz - margin));
            let cy = rng.gen_range(margin..(sz - margin));
            (cx, cy)
        } else if roll < 0.8 {
            // Edge: partially clipped.
            let cx = rng.gen_range(-half_side..(sz + half_side));
            let cy = rng.gen_range(-half_side..(sz + half_side));
            (cx, cy)
        } else {
            // Extreme: mostly off-canvas, just a corner or edge visible.
            let overshoot = half_side * 1.5;
            let cx = rng.gen_range(-overshoot..(sz + overshoot));
            let cy = rng.gen_range(-overshoot..(sz + overshoot));
            (cx, cy)
        };
        canvas.draw_rotated_square(cx, cy, half_side, angle);
    }

    Sample {
        pixels: canvas.pixels,
        num_circles: num_circles as f32,
        num_squares: num_squares as f32,
    }
}

/// Generate a batch of training data and return tensors on `device`.
///
/// Returns (images: [B,1,H,W], labels: [B,2]).
fn generate_batch<B: Backend>(
    batch_size: usize,
    rng: &mut StdRng,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 2>) {
    let mut all_pixels = Vec::with_capacity(batch_size * IMG_SIZE * IMG_SIZE);
    let mut all_labels = Vec::with_capacity(batch_size * 2);

    for _ in 0..batch_size {
        let sample = generate_sample(rng);
        all_pixels.extend_from_slice(&sample.pixels);
        all_labels.push(sample.num_circles);
        all_labels.push(sample.num_squares);
    }

    // Build tensors.  The data starts on the CPU (in `all_pixels`/`all_labels`)
    // and gets transferred to the GPU via `to_device`.  For burn-cuda this
    // is a single cudaMemcpyHostToDevice under the hood — one DMA transfer
    // over PCIe.  At batch=64 the images tensor is 64×1×64×64×4 bytes = 1 MB,
    // which saturates a PCIe 3.0 ×16 link in ~60 µs.
    let images = Tensor::<B, 1>::from_floats(all_pixels.as_slice(), device)
        .reshape([batch_size, 1, IMG_SIZE, IMG_SIZE]);

    let labels =
        Tensor::<B, 1>::from_floats(all_labels.as_slice(), device).reshape([batch_size, 2]);

    (images, labels)
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Training loop
// ═══════════════════════════════════════════════════════════════════════════

/// Training hyperparameters.
struct TrainConfig {
    num_epochs: usize,
    batches_per_epoch: usize,
    batch_size: usize,
    learning_rate: f64,
    checkpoint_every_epochs: usize,
    val_batches: usize,
    checkpoint_dir: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            num_epochs: 5000,
            batches_per_epoch: 300, // 300 × 64 = 19 200 samples/epoch
            batch_size: 64,
            learning_rate: 1e-3,
            checkpoint_every_epochs: 10,
            val_batches: 80, // 80 × 64 = 5 120 validation samples
            checkpoint_dir: "checkpoints".into(),
        }
    }
}

/// L1 (MAE) loss: mean of |pred - target| across all elements.
///
/// Unlike MSE which penalises large errors quadratically (biasing the
/// model toward predicting the distribution mean), L1 gives equal
/// gradient magnitude regardless of error size.  For counting tasks
/// this means the model tries equally hard to get "15 circles" right
/// as "3 circles".
fn l1_loss<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    (pred - target).abs().mean()
}

/// Run the full training procedure.
fn train<B: AutodiffBackend>(device: B::Device) {
    let cfg = TrainConfig::default();

    std::fs::create_dir_all(&cfg.checkpoint_dir).expect("failed to create checkpoint dir");

    let mut model = ShapeCounterConfig::new().init::<B>(&device);
    let mut optim = AdamConfig::new().init::<B, ShapeCounter<B>>();

    // Cosine annealing LR: starts at initial_lr, decays to min_lr over
    // total_iters steps following a half-cosine curve.
    let total_iters = cfg.num_epochs * cfg.batches_per_epoch;
    let mut lr_scheduler = burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig::new(
        cfg.learning_rate,
        total_iters,
    )
    .with_min_lr(1e-5)
    .init()
    .expect("failed to init LR scheduler");

    let mut train_rng = StdRng::seed_from_u64(42);
    let mut val_rng = StdRng::seed_from_u64(12345);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Shape Counter v2 — Training on CUDA                       ║");
    println!("║  Channels: 64→128→128→128  |  Loss: L1  |  LR: cosine     ║");
    println!(
        "║  Epochs: {:>4}  |  Batches/epoch: {:>4}  |  Batch size: {:>3} ║",
        cfg.num_epochs, cfg.batches_per_epoch, cfg.batch_size
    );
    println!(
        "║  LR: {:.0e} → 1e-5 (cosine)  |  Params: ~650 K             ║",
        cfg.learning_rate
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    for epoch in 1..=cfg.num_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0_f64;

        for batch_idx in 0..cfg.batches_per_epoch {
            let (images, labels) = generate_batch::<B>(cfg.batch_size, &mut train_rng, &device);

            // Intensity noise augmentation: multiply each pixel by a
            // random value in [0.7, 1.0].  Makes the model robust to
            // slight intensity differences between training and inference.
            let noise = Tensor::<B, 4>::random(
                images.dims(),
                burn::tensor::Distribution::Uniform(0.7, 1.0),
                &device,
            );
            let images = images * noise;

            let pred = model.forward(images);
            let loss = l1_loss(pred, labels);

            let loss_val: f64 = loss.clone().into_scalar().elem();
            epoch_loss += loss_val;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            let lr = burn::lr_scheduler::LrScheduler::step(&mut lr_scheduler);
            model = optim.step(lr, model, grads);

            if (batch_idx + 1) % 50 == 0 {
                println!(
                    "  [Epoch {:>3}/{} | Batch {:>4}/{}] loss = {:.6}  lr = {:.2e}",
                    epoch,
                    cfg.num_epochs,
                    batch_idx + 1,
                    cfg.batches_per_epoch,
                    loss_val,
                    lr
                );
            }
        }

        let avg_train_loss = epoch_loss / cfg.batches_per_epoch as f64;

        let val_model = model.valid();
        let mut val_loss_sum = 0.0_f64;
        let mut val_mae_circles = 0.0_f64;
        let mut val_mae_squares = 0.0_f64;

        for _ in 0..cfg.val_batches {
            let (images, labels) =
                generate_batch::<B::InnerBackend>(cfg.batch_size, &mut val_rng, &device);

            let pred = val_model.forward(images.clone());
            let loss = l1_loss(pred.clone(), labels.clone());
            val_loss_sum += loss.into_scalar().elem::<f64>();

            // Mean absolute error per output (more interpretable than MSE
            // for a counting task — "off by 0.3 circles on average").
            let diff = (pred - labels).abs();
            let mae: Vec<f32> = diff.mean_dim(0).into_data().to_vec().unwrap();
            val_mae_circles += mae[0] as f64;
            val_mae_squares += mae[1] as f64;
        }

        let avg_val_loss = val_loss_sum / cfg.val_batches as f64;
        let avg_mae_c = val_mae_circles / cfg.val_batches as f64;
        let avg_mae_s = val_mae_squares / cfg.val_batches as f64;
        let elapsed = epoch_start.elapsed();

        println!(
            "Epoch {:>3}/{} | train_loss: {:.6} | val_loss: {:.6} | \
             val_MAE(circles): {:.3} | val_MAE(squares): {:.3} | {:.1}s",
            epoch,
            cfg.num_epochs,
            avg_train_loss,
            avg_val_loss,
            avg_mae_c,
            avg_mae_s,
            elapsed.as_secs_f64()
        );

        // ── Checkpointing ─────────────────────────────────────────────
        if epoch % cfg.checkpoint_every_epochs == 0 || epoch == cfg.num_epochs {
            // ── Sample predictions dump ───────────────────────────────
            // Before saving, generate a small batch and show ground truth
            // vs predicted counts so you can eyeball how the model is
            // doing on real data.
            println!("  ┌─────────────────────────────────────────────────────────┐");
            println!("  │  Sample predictions (10 images)                        │");
            println!("  ├──────┬──────────────────┬──────────────────┬───────────┤");
            println!("  │  #   │  GT (circ, sq)    │  Pred (circ, sq) │   Error   │");
            println!("  ├──────┼──────────────────┼──────────────────┼───────────┤");

            let num_samples = 10;
            let mut sample_rng = StdRng::seed_from_u64(epoch as u64 * 9999);
            let (sample_imgs, sample_labels) =
                generate_batch::<B::InnerBackend>(num_samples, &mut sample_rng, &device);
            let sample_pred = val_model.forward(sample_imgs);

            // Pull predictions and labels back to CPU for printing.
            let pred_data: Vec<f32> = sample_pred.into_data().to_vec().unwrap();
            let label_data: Vec<f32> = sample_labels.into_data().to_vec().unwrap();

            for i in 0..num_samples {
                let gt_c = label_data[i * 2];
                let gt_s = label_data[i * 2 + 1];
                let pr_c = pred_data[i * 2];
                let pr_s = pred_data[i * 2 + 1];
                let err_c = (pr_c - gt_c).abs();
                let err_s = (pr_s - gt_s).abs();
                println!(
                    "  │  {:>2}  │  ({:>2.0}, {:>2.0})          │  ({:>5.2}, {:>5.2})   │ ({:.2},{:.2}) │",
                    i + 1, gt_c, gt_s, pr_c, pr_s, err_c, err_s
                );
            }
            println!("  └──────┴──────────────────┴──────────────────┴───────────┘");

            // ── Save checkpoint as pretty JSON ────────────────────────
            // PrettyJsonFileRecorder<FullPrecisionSettings> writes a
            // human-readable .json file with all parameter names as keys
            // and fp32 values.  File size is ~10 MB (vs ~1.7 MB for
            // CompactRecorder's half-precision msgpack), but you can
            // inspect/diff/grep it easily.
            let path = format!("{}/shape_counter_epoch_{}", cfg.checkpoint_dir, epoch);
            let recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
            model
                .clone()
                .save_file(&path, &recorder)
                .expect("failed to save checkpoint");
            println!("  → Checkpoint saved: {}.json", path);
        }
        println!();
    }

    let total_elapsed = total_start.elapsed();
    println!("════════════════════════════════════════════════════════════════");
    println!(
        "Training complete.  Total time: {:.1}s ({:.1} min)",
        total_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64() / 60.0
    );
    println!("════════════════════════════════════════════════════════════════");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Entry point
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    // Backend stack:
    //   Autodiff<Cuda>
    //
    // - `Cuda` is burn's native CUDA backend built on CubeCL.  It compiles
    //   tensor ops into PTX at runtime and launches them via the CUDA driver
    //   API.  Unlike the tch backend, there's no libtorch dependency — it's
    //   pure Rust all the way down to the PTX emission.
    //
    // - `Autodiff` wraps any backend with reverse-mode automatic
    //   differentiation.  It records a tape of operations during the forward
    //   pass, then replays it in reverse during `.backward()`.  The tape
    //   nodes store just enough info (tensor shapes, indices into a value
    //   arena) to reconstruct each gradient kernel — the actual activation
    //   tensors are reference-counted and freed eagerly once no longer needed
    //   by any live tape node.
    //
    // - Fusion is enabled by default for the Cuda backend (via the `fusion`
    //   feature flag).  This means consecutive elementwise ops (like BN's
    //   affine transform followed by ReLU) get fused into a single GPU
    //   kernel at JIT compile time.

    type TrainBackend = Autodiff<Cuda>;
    let device = CudaDevice::default();

    train::<TrainBackend>(device);
}
