#include "qmutils/expression.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/sparse_matrix.h"
#include "qmutils/term.h"

using namespace qmutils;

class HarmonicOscillator1D {
 public:
  HarmonicOscillator1D(float omega) : m_omega(omega) {}

  float omega() const { return m_omega; }

  Expression hamiltonian() const {
    const float hbar = 1.0f;

    Expression result;
    result += hbar * m_omega * Term::density(Operator::Spin::Up, 0);
    result += 0.5f * hbar * m_omega;
    return result;
  }

 private:
  float m_omega;  // frequency parameter
};

int main() {
  const size_t sites = 1;
  const size_t particles = 3;

  HarmonicOscillator1D oscillator(1.0f);
  Basis basis(sites, particles);

  for (const auto& elm : basis) {
    std::cout << Term(elm).to_string() << std::endl;
  }

  auto mat = compute_matrix_elements<SpMat_cf>(basis, oscillator.hamiltonian());

  for (size_t i = 0; i < basis.size(); ++i) {
    for (size_t j = 0; j < basis.size(); ++j) {
      std::cout << "H(" << i << "," << j << ") = " << mat(i, j) << std::endl;
    }
  }

  return 0;
}
