# Fast differentiable quantum circuit on GPU

This is an extension of [jax](https://github.com/google/jax) for quantum circuits differetiable programming. All the subroutines of the extension are written in Cuda C for better performance.

## Installation
You need to have CUDA compatible GPU and [CUDA](https://developer.nvidia.com/cuda-downloads) installed on your computer. Clone this repo on your local machine. Create a python virtual environment via
`python -m venv <name>` or use existing one. Activate the virtual environment. Install [maturin](https://maturin.rs/) in the virtual environment via `pip install maturin`. Go to the local directory with the cloned repo. Build the necessary dependencies by running `maturin develop --release` in the cloned repo directory for the signle precision arithmetic or `maturin develop --features "f64" --release` for the double precision arithmetic. Run `pip instal .` in the cloned repo directory to install the extension itself. Install numpy and jax in your virtual machine if they have not already been installed.

## Examples
In the root of the project one can find `example_vqse_ising.py`. A complete tutorial will be released asap.

## Issues with the current version of the package
1. One has to be careful with parallel execution of tests via cargo. Running `cargo test` leads to a race condition and incorrect results since different CPU threads send tasks to the GPU queue without synchronization. Use single-thread aliases `cargo t32` (`cargo t64`) for the single(double) precision tests execution.
2. The current version of the package interacts with jax via Rust [numpy crate](https://docs.rs/numpy/latest/numpy/).
Potentially, this could lead to bugs when one uses jit compilation. Do not use jit compilation with functions containing any components of a circuit, they are already optimized and fast. Nevertheless, one can combine those functions with jitted ones that do not include circuit components.
3. Currently, CUDA code is compiled with the following flag `-arch=sm_86` specifying a particular NVIDIA GPU architecture. If you want to use other flag, you can edit `build.rs` script in the root of the project. To figure out what flags you need, you can read the following [blogpost](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
4. Automatic differentiation relies on the unitarity of gates. If due to some reason one passes non-unitary gates, automatic differentiation breaks. This allows one to speed the code up and keep O(1) memory usage disregarding the circuit depth.