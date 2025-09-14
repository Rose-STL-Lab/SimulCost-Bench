# EPOCH Setup Guide for CostSci-Tools

This guide provides instructions for setting up EPOCH (Extendable PIC Open Collaboration) for use with the CostSci-Tools parameter optimization framework.

## Overview

EPOCH is a 1D Particle-in-Cell (PIC) code for laser-plasma interaction simulations. The CostSci-Tools integration requires:

- **Linux environment** (required for EPOCH compilation)
- **3 separate EPOCH binaries** compiled for different particle-weighting orders (2nd, 3rd, 5th)
- **Proper directory organization** within the solvers/ folder

## Automated Setup (Recommended)

### Prerequisites

Install the required dependencies:

```bash
sudo apt update
sudo apt install gfortran openmpi-bin libopenmpi-dev
```

### Run Setup Script

Execute the automated setup script from the repository root:

```bash
cd /path/to/costsci-tools
python solvers/setup_epoch.py
```

The script will automatically:

1. Initialize the existing EPOCH git submodule in `solvers/epoch/`
2. Configure Makefiles for different particle-weighting orders
3. Compile 3 separate binaries (2nd, 3rd, 5th order)
4. Update runner paths to use the new binary locations
5. Update physics table paths in input.deck
6. Verify all binaries are different

### Directory Structure After Setup

```
solvers/epoch/                    # Git submodule
├── epoch1d/                      # EPOCH 1D source code
│   ├── src/
│   │   └── physics_packages/
│   │       └── TABLES/           # Physics tables directory
│   ├── bin/                      # Binary directory
│   │   ├── epoch1d               # Latest compiled binary
│   │   ├── epoch1d_2nd           # 2nd order particle weighting
│   │   ├── epoch1d_3rd           # 3rd order particle weighting
│   │   └── epoch1d_5th           # 5th order particle weighting
│   └── Makefile                  # Modified for each compilation
```

## Manual Setup (For Reference)

If you need to set up EPOCH manually or troubleshoot the automated script:

### 1. Initialize EPOCH Submodule

Since EPOCH is already included as a git submodule in this repository:

```bash
git submodule update --init --recursive
cd solvers/epoch/epoch1d
```

### 2. Compile Different Particle Orders

**For 3rd Order (Default):**

```bash
make clean
make COMPILER=gfortran
cp bin/epoch1d bin/epoch1d_3rd
```

**For 2nd Order:**
Edit `Makefile` to uncomment:

```makefile
DEFINES += $(D)PARTICLE_SHAPE_TOPHAT
```

Then compile:

```bash
make clean
make COMPILER=gfortran
cp bin/epoch1d bin/epoch1d_2nd
```

**For 5th Order:**
Edit `Makefile` to uncomment:

```makefile
DEFINES += $(D)PARTICLE_SHAPE_BSPLINE3
```

Then compile:

```bash
make clean
make COMPILER=gfortran
cp bin/epoch1d bin/epoch1d_5th
```

### 3. Update Runner Paths

Edit `runners/epoch.py` to update binary paths:

```python
path_epoch2ndOrder = "solvers/epoch/epoch1d/bin/epoch1d_2nd"
path_epoch3rdOrder = "solvers/epoch/epoch1d/bin/epoch1d_3rd"
path_epoch5thOrder = "solvers/epoch/epoch1d/bin/epoch1d_5th"
```

### 4. Update Physics Table Path

Edit `runners/input.deck` line 40:

```
physics_table_location = solvers/epoch/epoch1d/src/physics_packages/TABLES/
```

### 5. Verify Setup

Check that all binaries are different:

```bash
cd solvers/epoch/epoch1d/bin
cmp epoch1d_2nd epoch1d_3rd && echo "ERROR: 2nd and 3rd are identical" || echo "✅ 2nd and 3rd differ"
cmp epoch1d_2nd epoch1d_5th && echo "ERROR: 2nd and 5th are identical" || echo "✅ 2nd and 5th differ"
cmp epoch1d_3rd epoch1d_5th && echo "ERROR: 3rd and 5th are identical" || echo "✅ 3rd and 5th differ"
```

## Usage

Once set up, EPOCH simulations will automatically use the appropriate binary based on the `particle_order` parameter:

- `particle_order: 2` → uses `solvers/epoch/epoch1d/bin/epoch1d_2nd`
- `particle_order: 3` → uses `solvers/epoch/epoch1d/bin/epoch1d_3rd`
- `particle_order: 5` → uses `solvers/epoch/epoch1d/bin/epoch1d_5th`

## Troubleshooting

### Compilation Errors

- Ensure gfortran and OpenMPI are properly installed
- Check that you're on a Linux system (EPOCH requires Linux)
- Verify git submodule was initialized correctly

### Binary Issues

- If binaries are identical, the Makefile modifications didn't work
- Re-run the setup script or manually edit Makefiles
- Ensure `make clean` was run between compilations

### Runtime Errors

- Check that binary paths in `runners/epoch.py` are correct
- Verify physics table path in `runners/input.deck`
- Ensure binaries have execute permissions (`chmod +x`)

## Parameter Optimization

EPOCH supports optimization of 5 parameters:

- **`dt_multipler`**: Controls temporal discretization [0.80-0.99]
- **`nx`**: Spatial grid resolution [400→]
- **`npart`**: Particles per cell [10→]
- **`field_order`**: Field integration order {2,4,6}
- **`particle_order`**: Particle weighting order {2,3,5}

The automated dummy solutions will explore these parameter spaces across 3 precision levels and 3 physics profiles.
