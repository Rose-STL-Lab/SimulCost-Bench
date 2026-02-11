FROM python:3.11-bookworm

# ── System dependencies ─────────────────────────────────────────────
# EPOCH: gfortran + OpenMPI
# Euler 2D: cmake + build-essential + Eigen3 + TBB
# FEM 2D: g++ (included in build-essential)
# General: git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gfortran \
        openmpi-bin libopenmpi-dev \
        cmake build-essential \
        libeigen3-dev libtbb-dev \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Poetry & configure ──────────────────────────────────────
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# ── Dependency layer caching ────────────────────────────────────────
# Copy only dependency manifests first so code changes don't trigger reinstall
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction

# ── Copy project source ────────────────────────────────────────────
COPY . .

# ── Compile EPOCH (2nd, 3rd, 5th order binaries) ───────────────────
RUN <<'BASH'
set -eux

# 2nd
python - <<'PY'
import re
import pathlib

mf = pathlib.Path("costsci_tools/solvers/epoch/epoch1d/Makefile")

def modify(order: str) -> None:
    c = mf.read_text()
    # comment out both
    c = re.sub(r'^(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_TOPHAT.*)$',  r'#\1', c, flags=re.MULTILINE)
    c = re.sub(r'^(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_BSPLINE3.*)$', r'#\1', c, flags=re.MULTILINE)

    # enable chosen
    if order == "2nd":
        c = re.sub(r'^#(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_TOPHAT.*)$',  r'\1', c, flags=re.MULTILINE)
    elif order == "5th":
        c = re.sub(r'^#(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_BSPLINE3.*)$', r'\1', c, flags=re.MULTILINE)
    elif order == "3rd":
        pass
    else:
        raise ValueError(order)

    mf.write_text(c)

modify("2nd")
PY
make COMPILER=gfortran -C costsci_tools/solvers/epoch/epoch1d
cp costsci_tools/solvers/epoch/epoch1d/bin/epoch1d costsci_tools/solvers/epoch/epoch1d/bin/epoch1d_2nd
make clean -C costsci_tools/solvers/epoch/epoch1d

# 3rd
python - <<'PY'
import re
import pathlib

mf = pathlib.Path("costsci_tools/solvers/epoch/epoch1d/Makefile")

def modify(order: str) -> None:
    c = mf.read_text()
    c = re.sub(r'^(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_TOPHAT.*)$',  r'#\1', c, flags=re.MULTILINE)
    c = re.sub(r'^(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_BSPLINE3.*)$', r'#\1', c, flags=re.MULTILINE)

    if order == "2nd":
        c = re.sub(r'^#(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_TOPHAT.*)$',  r'\1', c, flags=re.MULTILINE)
    elif order == "5th":
        c = re.sub(r'^#(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_BSPLINE3.*)$', r'\1', c, flags=re.MULTILINE)
    elif order == "3rd":
        pass
    else:
        raise ValueError(order)

    mf.write_text(c)

modify("3rd")
PY
make COMPILER=gfortran -C costsci_tools/solvers/epoch/epoch1d
cp costsci_tools/solvers/epoch/epoch1d/bin/epoch1d costsci_tools/solvers/epoch/epoch1d/bin/epoch1d_3rd
make clean -C costsci_tools/solvers/epoch/epoch1d

# 5th
python - <<'PY'
import re
import pathlib

mf = pathlib.Path("costsci_tools/solvers/epoch/epoch1d/Makefile")

def modify(order: str) -> None:
    c = mf.read_text()
    c = re.sub(r'^(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_TOPHAT.*)$',  r'#\1', c, flags=re.MULTILINE)
    c = re.sub(r'^(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_BSPLINE3.*)$', r'#\1', c, flags=re.MULTILINE)

    if order == "2nd":
        c = re.sub(r'^#(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_TOPHAT.*)$',  r'\1', c, flags=re.MULTILINE)
    elif order == "5th":
        c = re.sub(r'^#(\s*DEFINES\s*\+=\s*\$\(D\)PARTICLE_SHAPE_BSPLINE3.*)$', r'\1', c, flags=re.MULTILINE)
    elif order == "3rd":
        pass
    else:
        raise ValueError(order)

    mf.write_text(c)

modify("5th")
PY
make COMPILER=gfortran -C costsci_tools/solvers/epoch/epoch1d
cp costsci_tools/solvers/epoch/epoch1d/bin/epoch1d costsci_tools/solvers/epoch/epoch1d/bin/epoch1d_5th
make clean -C costsci_tools/solvers/epoch/epoch1d

BASH

# ── Compile Euler 2D ────────────────────────────────────────────────
RUN mkdir -p costsci_tools/solvers/euler_2d_utils/CSMPM_BOW/build && \
    cd costsci_tools/solvers/euler_2d_utils/CSMPM_BOW/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j"$(nproc)"

# ── Compile FEM 2D (FastIPC wrapper) ────────────────────────────────
RUN cd costsci_tools/solvers/fastipc_utils/common/math/wrapper && \
    g++ -shared -fPIC -mavx -mfma -mavx2 \
        -I. -I./Eigen -I./EVCTCD \
        -o a.so wrapper.cpp EVCTCD/CTCD.cpp

# ── Update input.deck physics_table_location to container path ──────
RUN sed -i 's|physics_table_location\s*=\s*.*|physics_table_location = /app/costsci_tools/solvers/epoch/epoch1d/src/physics_packages/TABLES/|' \
    costsci_tools/runners/input.deck

# ── Environment variables ──────────────────────────────────────────
ENV PYTHONPATH=/app
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV MPLBACKEND=Agg

# ── Declare output volumes ─────────────────────────────────────────
VOLUME ["/app/sim_res", "/app/eval_results", "/app/results_model_attempt", "/app/log_model_tool_call"]

CMD ["bash"]
