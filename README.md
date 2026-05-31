# TinyLKF 🚀
**A lightweight and zero-allocation Linear Kalman Filter for microcontrollers.**

Using a Python-based symbolic algebra generator (powered by SymPy), TinyLKF resolves all matrix multiplications, inversions, and sparsity optimizations *before* compilation. It outputs raw, loop-free, and matrix-free C++ code tailored exactly to your physical model.

## ✨ Features
* **Zero Matrix Math on MCU:** Matrix inversions and multiplications are solved algebraically in Python. The MCU only executes raw floating-point arithmetic.
* **Zero Loops (Loop Unrolling):** No `for` or `while` loops. The code is flattened for maximum execution speed.
* **Zero Dynamic Allocation:** No `new` or `malloc()`. Memory is statically allocated at compile time.
* **Framework Agnostic:** Generates standard C++ code. .

1. **The Generator (Python):** You define your State-Space model (Matrices $F$, $B$, $H$) in a Jupyter Notebook. The script runs symbolic algebra, cuts out mathematical redundancies (like multiplying by zero), and handles the Kalman Gain matrix inversion.
2. **The Target (C++):** The script auto-generates `TinyLKF.h` and `TinyLKF.cpp` files. You simply drop these into your MCU project folder.