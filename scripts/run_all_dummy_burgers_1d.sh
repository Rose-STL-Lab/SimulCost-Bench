#!/bin/bash

echo "Running 1D_burgers experiments..."

python qs_gen/1D_burgers.py -t cfl -z
python qs_gen/1D_burgers.py -t cfl
python qs_gen/1D_burgers.py -t k -z
python qs_gen/1D_burgers.py -t k
python qs_gen/1D_burgers.py -t w -z
python qs_gen/1D_burgers.py -t w

echo "Done."
