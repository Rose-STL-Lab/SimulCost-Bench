# Euler 2D Setup Guide for CostSci-Tools

This guide provides instructions for setting up the CSMPM_BOW Euler 2D gas dynamics solver for use with the CostSci-Tools parameter optimization framework.

## Overview

CSMPM_BOW is a 2D Compressible Euler gas dynamics solver implemented in C++. The CostSci-Tools integration requires:

- **Linux environment** (recommended for compilation)
- **Compiled gas_2d binary** from the CSMPM_BOW codebase
- **Proper directory organization** within the solvers/ folder

## Automated Setup (Recommended)

### Prerequisites

Install the required dependencies:

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libeigen3-dev libtbb-dev python3-dev python3-numpy python3-matplotlib
```

### Run Setup Script

Execute the automated setup script from the repository root:

```bash
cd /path/to/costsci-tools
python solvers/setup_euler_2d.py
```

The script will automatically:

1. Check for required dependencies (CMake, Eigen3, TBB, Python)
2. Navigate to the CSMPM_BOW directory
3. Create build directory and configure with CMake
4. Compile the gas_2d binary
5. Verify the binary was created successfully
6. Print the binary location for wrapper use

### Directory Structure After Setup

```
solvers/euler_2d_utils/
└── CSMPM_BOW/                    # Standalone C++ solver
    ├── build/                    # Build directory
    │   └── Examples/
    │       └── gas_2d            # Compiled binary (~580 KB)
    ├── Examples/
    │   └── gas_2d.cpp           # 2D gas simulation source
    ├── Libs/                    # Core libraries
    │   ├── BowReplacement/      # BOW framework replacements
    │   ├── EOS/                 # Equation of state
    │   ├── EulerGas/            # Euler gas operators
    │   ├── RPSolver/            # WENO reconstruction
    │   ├── TimeIntegration/     # TVD Runge-Kutta
    │   ├── LinearProjectionSys/ # Projection methods
    │   ├── Simulator/           # Gas simulator
    │   └── IO/                  # VTK output
    ├── CMakeLists.txt           # Build configuration
    └── README.md                # Project overview
```

## Manual Setup (For Reference)

If you need to set up the solver manually or troubleshoot the automated script:

### 1. Install Dependencies

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libeigen3-dev libtbb-dev python3-dev python3-numpy python3-matplotlib
```

### 2. Navigate to CSMPM_BOW Directory

```bash
cd solvers/euler_2d_utils/CSMPM_BOW
```

### 3. Build the Project

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake (Release mode for performance)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile
make -j$(nproc)

# Verify binary was created
ls -lh Examples/gas_2d
```

Expected output: `Examples/gas_2d` binary (~580 KB)

### 4. Test the Binary

```bash
# Run test case 0 (central explosion) with small grid for quick test
./Examples/gas_2d 0 0 5 16 0.075 0.25 1e-7 ./tmp/test_euler
```

Expected output:

- Log messages showing simulation progress
- Creates directory `./tmp/test_euler/`
- Generates VTK files in `./tmp/test_euler/vtk/`
- Creates `meta.json` with simulation metadata

## Usage

Once set up, the gas_2d binary is invoked through the Python wrapper interface. The binary accepts the following command-line arguments:

```bash
./gas_2d <testcase> <start_frame> <end_frame> <N_grid_x> <record_dt> <cfl> <cg_tolerance> <output_dir>
```

**Parameters:**

- `testcase`: Test case number (0-8)
- `start_frame`: Starting frame number
- `end_frame`: Ending frame number
- `N_grid_x`: Grid resolution in x-direction
- `record_dt`: Time interval between output frames
- `cfl`: CFL number for timestep stability
- `cg_tolerance`: CG solver convergence tolerance
- `output_dir`: Directory for output files

**Output:**

- `<output_dir>/vtk/` - VTK structured grid files (one per frame)
- `<output_dir>/meta.json` - Simulation metadata (cost, parameters, runtime)

## Troubleshooting

### Compilation Errors

**Eigen3 Not Found:**

```bash
# Manually specify Eigen3 path
cmake .. -DEigen3_DIR=/usr/share/eigen3/cmake
```

**TBB Not Found:**

```bash
# Ensure TBB is installed
sudo apt-get install libtbb-dev
```

**C++17 Support:**

```bash
# Check compiler version (need GCC >= 7.0)
g++ --version
```

### Runtime Errors

**Binary Not Found:**

```bash
# Verify binary exists
ls -lh solvers/euler_2d_utils/CSMPM_BOW/build/Examples/gas_2d
```

**Output Directory Errors:**

- Ensure write permissions in the working directory
- Check available disk space: `df -h`

**Segmentation Faults:**

- Try smaller grid resolution (e.g., n_grid_x=16 or 32)
- Check memory availability
- Rebuild in Debug mode for more info: `cmake .. -DCMAKE_BUILD_TYPE=Debug`

### Performance Issues

- Use **Release** build for production runs (10-100x faster than Debug)
- Start with small grids (e.g., 32×32) for testing
- Increase grid resolution gradually
- Monitor memory usage (scales with `N_grid_x^2`)

**Note**: This solver was originally part of the BOW physics framework and has been refactored as a standalone project for easier integration.
