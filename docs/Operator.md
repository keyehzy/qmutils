# qmutils::Operator - Representing Quantum Mechanical Operators

The `qmutils::Operator` class is a fundamental building block for representing
second-quantized fermionic operators commonly used in quantum many-body physics,
such as creation and annihilation operators. It provides a compact and efficient
way to store and manipulate these operators, forming the basis for more complex
structures like `Term` and `Expression`.

## Rationale and Design

In second quantization, fermionic systems are described by creation
(c<sup>†</sup>) and annihilation (c) operators acting on specific quantum
states. These states are typically characterized by their spin (up ↑ or down ↓)
and orbital index (representing spatial or other degrees of freedom). The
`Operator` class encapsulates this information into a single, small object.

**Compact Representation:**

The design prioritizes minimal memory footprint. It uses bit fields within a
single byte to represent the operator's type, spin, and orbital index. This
compactness is crucial for performance when dealing with large expressions
involving many operators. Specifically:

* **Type:** 1 bit (0 for creation, 1 for annihilation)
* **Spin:** 1 bit (0 for spin up, 1 for spin down)
* **Orbital:** 6 bits (allowing for up to 64 orbitals)

This design allows for efficient storage and manipulation of operators,
minimizing overhead and maximizing performance, particularly in memory-bound
operations.

## Usage

Here is a demonstratinon how to create and use `Operator` objects:

```c++
#include "qmutils/operator.h"

int main() {
  using namespace qmutils;

  // Create a creation operator for spin-up electron in orbital 5
  Operator c_up_5 = Operator::creation(Operator::Spin::Up, 5);

  // Create an annihilation operator for spin-down electron in orbital 3
  Operator a_down_3 = Operator::annihilation(Operator::Spin::Down, 3);

  // Access operator properties
  Operator::Type type = c_up_5.type();   // Operator::Type::Creation
  Operator::Spin spin = a_down_3.spin(); // Operator::Spin::Down
  uint8_t orbital = c_up_5.orbital();    // 5

  // Check commutation relation
  bool commute = c_up_5.commutes_with(a_down_3); // True (different spin/orbital)

  // Obtain the adjoint operator
  Operator a_up_5 = c_up_5.adjoint();       // Annihilation operator for spin up, orbital 5

  // Flip the spin
  Operator c_down_5 = c_up_5.flip_spin();  // Creation operator for spin down, orbital 5

  return 0;
}
```
