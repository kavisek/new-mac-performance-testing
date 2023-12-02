# new-mac-performance-testing

This is a repo on performance testing a back for python and machine learning development. Contains some simple python scripts.

- Python Matrix Multiplication
- GPU Benchmarking (Pytorch) (Notebook)

```bash
# This example uses python 3.11
cd app 
poetry install
poetry shell
python ./scripts/matrix_multiplication.py
```

## 2023 M3 Max Macbook Pro

Results: Matrix multiplication of size 16000x16000 took 16.08 seconds

### View Core Utilization

1. Open Activity Monitor
2. Right Click the Activity Monitor Dock Icon
3. Select Monitor -> Show CPU History

![cores](./images/cores.png){ width=25% }

### View GCP Utilization

1. Open Activity Monitor
2. Select Window in the Menu Bar -> GPU History

![gpu](./images/gpu.png)

### Machine Learning Performance (Pytorch Matrix Multiplication)

You can view the nootebook [here](./app/nootebooks/gpu_benchmark.ipynb). This notebook runs 30 samples of matrix multiplication of size 50,0000x50,0000 tensor on both the CPU and GPU. The GPU is ~10x faster than the CPU for machine learning workloads using Pytorch.

I am currently using a 2023 Macbook Pro with a M1 Max Utra Configuration with 16 core CPU and 40 core GPU and 128GB of unified memory (RAM).




```bash
cd app
poetry install
poetry run jupyter lab 
```

Here are the notebook results:

![gpu](./images/gpu_benchmark.png)


### References

- https://github.com/alexziskind1/machine_tests/tree/main/python/matrix_multiplication
- https://github.com/PlummersSoftwareLLC/Primes
- https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/mandelbrot-python3-7.html