# FastIPC Solver Setup Guide

This guide provides instructions for setting up the FastIPC solver for use with the CostSci-Tools parameter optimization framework.

## Overview

The FastIPC solver is a C++/Taichi hybrid solver for simulating deformable objects with contact and friction. It implements implicit finite element methods with corotational elasticity for nonlinear solid mechanics simulations.

The CostSci-Tools integration uses FastIPC as a git submodule and requires:

- A C++ compiler (e.g., g++)
- CPU with AVX, AVX2, and FMA support
- A compiled shared library `a.so` from the C++ source code

## Quick Setup (Recommended)

The easiest way to set up FEM2D is to use the automated setup script:

```bash
cd costsci_tools
python solvers/setup_fem2d.py
```

This script will:
1. Initialize the FastIPC git submodule
2. Check for required dependencies
3. Compile the C++ shared library with all necessary source files
4. Verify the library loads correctly

## Manual Setup

If you prefer to set up manually or the automated script fails, follow these steps:

### Prerequisites

Install the required dependencies:

```bash
sudo apt update
sudo apt install g++ git
```

Verify your CPU supports the required instruction sets:

```bash
lscpu | grep -E 'avx|avx2|fma'
```

You should see output indicating AVX, AVX2, and FMA support. If not, the compiled library may not work on your system.

### Initialize Submodule

From the repository root:

```bash
git submodule update --init --recursive
```

This will initialize the `fastipc_utils` submodule.

### Compilation

Navigate to the wrapper directory and compile the shared library:

```bash
cd solvers/fastipc_utils/common/math/wrapper
g++ -shared -fPIC -mavx -mfma -mavx2 -I. -I./Eigen -I./EVCTCD -o a.so wrapper.cpp EVCTCD/CTCD.cpp
```

**Important:** Make sure to include both `wrapper.cpp` and `EVCTCD/CTCD.cpp` in the compilation command, otherwise you'll get undefined symbol errors.

This will create a shared library file `a.so` in the same directory (typically ~17 MB).

### Verification

Verify the library was compiled correctly:

```bash
ls -lh solvers/fastipc_utils/common/math/wrapper/a.so
```

You should see a file of approximately 17 MB.

## Running FEM2D Simulations

Once the C++ library is compiled, the FEM2D solver can be run from the repository root:

```bash
python runners/fem2d.py
```

The runner automatically sets up the Python path to include both the repository root and the `fastipc_utils` directory.

### Test Cases

FEM2D includes three test cases:

- **p1 (cantilever):** Beam bending under gravity with large deformation
  ```bash
  python runners/fem2d.py --config-name=p1
  ```

- **p2 (vibration_bar):** 1D elastic wave propagation and compression dynamics
  ```bash
  python runners/fem2d.py --config-name=p2
  ```

- **p3 (twisting_column):** 2D rotational dynamics and energy conservation
  ```bash
  python runners/fem2d.py --config-name=p3
  ```

### Parameter Tuning

Override parameters via command line:

```bash
python runners/fem2d.py nx=40 dt=0.00025 newton_v_res_tol=0.005
```

Key tunable parameters:
- `nx`: Grid resolution (number of cells in x-direction)
- `dt`: Time step size
- `newton_v_res_tol`: Newton solver convergence tolerance

## Troubleshooting

### "No module named 'common'" Error

This means the Python path is not set correctly. The runner should automatically add the necessary paths, but if you're running the solver from a different location, make sure to add:

```python
import sys
import os
repo_root = os.path.abspath('/path/to/costscit-tools2')
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, 'solvers', 'fastipc_utils'))
```

### "undefined symbol" Error

This typically means the shared library was compiled without all necessary source files. Make sure to include `EVCTCD/CTCD.cpp` in the compilation command:

```bash
g++ -shared -fPIC -mavx -mfma -mavx2 -I. -I./Eigen -I./EVCTCD -o a.so wrapper.cpp EVCTCD/CTCD.cpp
```

### CPU Doesn't Support AVX/AVX2/FMA

If your CPU doesn't support these instruction sets, you may need to compile without these flags (though performance will be reduced):

```bash
g++ -shared -fPIC -I. -I./Eigen -I./EVCTCD -o a.so wrapper.cpp EVCTCD/CTCD.cpp
```

### Library Won't Load

Check that the library file exists and has the correct permissions:

```bash
ls -lh solvers/fastipc_utils/common/math/wrapper/a.so
chmod +x solvers/fastipc_utils/common/math/wrapper/a.so
```

> **Using Docker?** These solvers are pre-compiled in the Docker image — no manual setup needed.
