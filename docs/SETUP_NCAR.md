# Running the OpenVLA Instruction-Swap Experiment on NCAR

End-to-end setup guide for NCAR supercomputer (Linux, NVIDIA H100/A100 80GB).

## 1. Clone the repo

```bash
git clone https://github.com/ktwu01/Gibbs-VLA-Overshoot.git
cd Gibbs-VLA-Overshoot
```

## 2. Create a conda environment

NCAR typically uses conda (via `module load`). If conda is available:

```bash
module load conda        # or: module load anaconda
conda create -n gibbs python=3.11 -y
conda activate gibbs
```

If no conda, use a venv:

```bash
module load python/3.11  # check: module avail python
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install PyTorch with CUDA

Check your CUDA version first:

```bash
nvidia-smi   # look at "CUDA Version" in the top right
```

Then install PyTorch matching that CUDA version:

```bash
# For CUDA 12.x (most common on H100/A100):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU access:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA H100 80GB HBM3  (or similar)
```

## 4. Install the project

```bash
pip install -e ".[openvla]"
```

This installs: transformers 4.40-4.49, timm <1.0, accelerate, Pillow, plus the analysis/models packages.

## 5. HuggingFace authentication (optional but recommended)

OpenVLA weights are public, but authentication speeds up downloads and avoids rate limits:

```bash
pip install huggingface-hub
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

## 6. Run the experiments

### 6a. Synthetic Gibbs experiment (quick sanity check, no GPU needed)

```bash
python -m experiments.synthetic_step \
    --output experiments/outputs/synthetic_gibbs_analysis.png
```

Expected: ~9.05% overshoot (the Gibbs constant).

### 6b. Mock instruction-swap experiment (no GPU needed)

```bash
python -m experiments.instruction_swap_inference \
    --output experiments/outputs/instruction_swap.png \
    --total-steps 100 --switch-step 50
```

### 6c. OpenVLA + Bridge V2 instruction-swap experiment (GPU required)

This is the real experiment. First run downloads:
- OpenVLA-7B weights (~15GB) from HuggingFace
- Uses a synthetic Bridge V2 observation (real TFDS Bridge V2 loading is optional)

```bash
python -m experiments.openvla_swap_experiment \
    --output experiments/outputs/openvla_swap.png \
    --total-steps 100 \
    --switch-step 50
```

The script auto-detects CUDA and loads the model in bfloat16 (~14GB VRAM).

**Expected runtime:** ~2 minutes on H100 (model load ~30s, 100 inference steps ~60s).

**Expected output:**
- `experiments/outputs/openvla_swap.png` — 6-panel diagnostic figure
- `experiments/outputs/openvla_swap.json` — metrics (overshoot %, PSD slope, etc.)

### Custom instruction pairs

```bash
python -m experiments.openvla_swap_experiment \
    --output experiments/outputs/openvla_swap_custom.png \
    --total-steps 100 --switch-step 50 \
    --instruction-a "pick up the corn" \
    --instruction-b "close the drawer"
```

### Raw (normalized) actions (no de-normalization)

```bash
python -m experiments.openvla_swap_experiment \
    --output experiments/outputs/openvla_swap_raw.png \
    --total-steps 100 --switch-step 50 \
    --unnorm-key none
```

## 7. SLURM batch job (if interactive nodes are limited)

Create `run_experiment.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=gibbs-openvla
#SBATCH --partition=gpu          # adjust to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

module load conda
conda activate gibbs

cd $HOME/Gibbs-VLA-Overshoot

python -m experiments.openvla_swap_experiment \
    --output experiments/outputs/openvla_swap.png \
    --total-steps 100 --switch-step 50

echo "Done. Results in experiments/outputs/"
```

Submit:

```bash
sbatch run_experiment.slurm
```

## 8. Verify results

After the experiment completes:

```bash
cat experiments/outputs/openvla_swap.json | python -m json.tool
```

Key metrics to check:
- `overshoot_percent`: How close to 8.95% (Gibbs constant)?
- `power_law_exponent`: Near -1.0 indicates step-like transient
- `per_axis_overshoot_percent`: Overshoot on each of the 7 action dims

## Troubleshooting

**`ImportError: timm version must be >= 0.9.10 and < 1.0.0`**
```bash
pip install "timm>=0.9.10,<1"
```

**`RuntimeError: causal mask size mismatch`**
The code patches this automatically. If it still occurs, ensure transformers <4.50:
```bash
pip install "transformers>=4.40,<4.50"
```

**`CUDA out of memory`**
Shouldn't happen on 80GB, but if running multi-GPU jobs on a shared node:
```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.openvla_swap_experiment ...
```

**Model download hangs/fails**
Set HF cache to a writable location with enough space:
```bash
export HF_HOME=/glade/work/$USER/.cache/huggingface  # adjust for your filesystem
```
