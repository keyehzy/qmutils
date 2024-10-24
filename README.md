# qmutils

[![Passing](https://github.com/keyehzy/qmutils/workflows/CI/badge.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

qmutils is a fast, modern C++ library for symbolic quantum mechanics calculations, focusing on fermionic systems. It provides an intuitive interface for quantum operator algebra, including efficient normal ordering and matrix element computation, while maintaining high computational performance.

## Why Use qmutils?

- **Fast:** Optimized implementations with caching and parallel computation support dramatically improve performance, especially for complex expressions and repeated operations.  Benchmarks show significant speedup compared to naive implementations.
- **Intuitive:** Clean, modern C++ interface for defining and manipulating quantum operators and expressions.
- **Flexible:** Works with any quantum many-body system, allowing for the study of diverse models like the Hubbard model, t-J model, and other lattice models.
- **Reliable:** Comprehensive test coverage and benchmarks ensure the accuracy and reliability of the library.

## Quick Start

### Installation

```bash
git clone https://github.com/organization/qmutils.git
cd qmutils
mkdir build && cd build
cmake ..
make
```

### Basic Usage

```cpp
#include "qmutils/basis.h"
#include "qmutils/expression.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/operator.h"
#include "qmutils/sparse_matrix.h"

using namespace qmutils;

// Create a simple Hamiltonian
const float t = 1.0f;
const float u = 0.5f;
Expression H;
H += t * Term::hopping(0, 1, Operator::Spin::Up);   // Hopping term
H += u * Term::density(Operator::Spin::Up, 0);      // Density term

// Work with basis states
Basis basis(4, 2);  // 4 orbitals, 2 particles
auto matrix = compute_matrix_elements<arma::cx_fmat>(basis, H);
```

See [examples](examples/) for more complex usages.

## Requirements

- C++17 compiler (GCC ≥ 7.0, Clang ≥ 5.0, or MSVC ≥ 19.14)
- CMake ≥ 3.10

Optional:
- Google Test (for testing)
- Google Benchmark (for benchmarks)
- Armadillo (for examples)
- OpenMP


## Features
- Creation/annihilation operators with spin and orbital indices.
- Efficient symbolic normal ordering with caching.
- Basis state management.
- Matrix element computation (dense and sparse matrix support).
- Fourier transformations of operators and expressions.
- Utilities for multi-dimensional lattice indexing.


## Documentation

- [Examples](examples/)
- [Docs](docs/)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citing qmutils

If you use qmutils in your research, please cite:

```bibtex
@software{qmutils2024,
  author = {de Sousa, M.},
  title = {qmutils: A C++ Library for Quantum Many-Body Calculations},
  year = {2024},
  url = {https://github.com/keyehzy/qmutils}
}
```
