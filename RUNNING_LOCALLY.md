# Running Newton locally (Apple Silicon)

Verified working on an M-series Mac (arm64) on 2026-07-10.

## Setup

```bash
uv sync --extra examples
```

## Run a simulation with the interactive GUI

macOS has no NVIDIA GPU, so Newton runs on CPU; the default `gl` viewer
(an OpenGL window) still works fine for visualization.

```bash
uv run -m newton.examples basic_pendulum --viewer gl
```

This opens a live OpenGL window showing the simulation. It's interactive and
keeps running until you close the window — it does not exit on its own after
a fixed number of frames (`--num-frames` only applies to the non-interactive
`usd`/`rtx` viewers).

Confirmed: the window opens, the process runs continuously without errors,
and the pendulum animates in real time.

## Other useful viewer options

```bash
# Force CPU device explicitly
uv run -m newton.examples basic_pendulum --viewer gl --device cpu

# Headless run that writes a USD file instead of opening a window
uv run -m newton.examples basic_pendulum --viewer usd --output-path my_output.usd --num-frames 300

# Browse all examples from one window
uv run -m newton.examples
```

## Notes

- First run triggers Warp kernel compilation/caching, so startup may take a
  few seconds longer than subsequent runs.
- No CUDA toolkit or GPU driver setup is needed on macOS.
