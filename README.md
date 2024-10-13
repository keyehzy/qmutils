# qmutils: C++ Quantum Mechanics Library

## Overview

qmutils is a C++ library designed for efficient quantum mechanics calculations.
It provides a set of tools for creating and manipulating quantum operators,
constructing Hamiltonians, and performing various quantum mechanical
computations.

### Key Features

- Efficient representation of quantum operators
- Support for creating complex quantum expressions
- Tools for constructing and solving common quantum models (e.g., Hubbard model)
- High-performance calculations optimized for large systems

## Installation

### Prerequisites

- C++17 compatible compiler
- CMake (version 3.10 or higher)
- Boost (for boost::container::small_vector)
- Google Test (for running tests)
- Google Benchmark (for running benchmarks)

### Building the Library

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/qmutils.git
   cd qmutils
   ```

2. Create a build directory and run CMake:
   ```
   mkdir build && cd build
   cmake ..
   ```

3. Build the library:
   ```
   make
   ```

## Usage

Here's a basic example of how to use qmutils to create a simple quantum system:

```cpp
#include "qmutils/operator.h"
#include "qmutils/term.h"

int main() {
    // Create creation and annihilation operators
    auto c_up = qmutils::Operator::creation(qmutils::Operator::Spin::Up, 0);
    auto c_down = qmutils::Operator::creation(qmutils::Operator::Spin::Down, 0);

    // Create a term representing the density of up-spin particles
    auto n_up = qmutils::Term(1.0, {c_up, c_up.adjoint()});

    // Print the term
    std::cout << n_up.to_string() << std::endl;

    return 0;
}
```

For more detailed examples, including how to construct Hamiltonians and perform
calculations, please refer to the `examples` directory in the repository.

## Testing and Benchmarking

qmutils uses Google Test for unit testing and Google Benchmark for performance
benchmarking. To run the tests and benchmarks:

1. Ensure you've built the project with the testing and benchmarking options enabled:
   ```
   mkdir build && cd build
   cmake -DBUILD_TESTING=ON -DBUILD_BENCHMARK=ON ..
   make
   ```

2. Run the tests:
   ```
   ./tests/qmutils-test
   ```

3. Run the benchmarks:
   ```
   ./benchmark/qmutils_benchmark
   ```

## License

qmutils is released under the MIT License. See the LICENSE file for more
details.