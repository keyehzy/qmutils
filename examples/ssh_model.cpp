#include <armadillo>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "qmutils/basis.h"
#include "qmutils/expression.h"
#include "qmutils/functional.h"
#include "qmutils/index.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/operator.h"

using namespace qmutils;

template <typename IndexType>
class SSHModel {
 public:
  SSHModel(float t1, float t2, float delta, IndexType index)
      : m_t1(t1), m_t2(t2), m_delta(delta), m_index(index) {}

  Expression hamiltonian() const {
    Expression H;

    auto hopping = [this](size_t unit_cell_1, size_t size_1, size_t unit_cell_2,
                          size_t site_2, float t) {
      Expression term;
      uint8_t orbital1 = m_index.to_orbital(unit_cell_1, size_1);
      uint8_t orbital2 = m_index.to_orbital(unit_cell_2, site_2);
      term += t * Expression::hopping(orbital1, orbital2, Operator::Spin::Up);
      term += t * Expression::hopping(orbital1, orbital2, Operator::Spin::Down);
      return term;
    };

    const size_t L = m_index.dimension(0);

    // Intra-cell hopping
    for (uint8_t i = 0; i < L; ++i) {
      H += hopping(i, 0, i, 1, m_t1 + m_delta);
    }

    // Inter-cell hopping
    for (uint8_t i = 0; i < L; ++i) {
      H += hopping(i, 1, (i + 1) % L, 0, m_t2 - m_delta);
    }

    return H;
  }

  const IndexType& index() const { return m_index; }

 private:
  float m_t1, m_t2, m_delta;
  IndexType m_index;
};

template <size_t L>
Expression fourier_transform_operator(const Operator& op,
                                      const StaticIndex<L, 2>& lattice) {
  Expression result;
  const float type_sign =
      (op.type() == Operator::Type::Annihilation) ? -1.0f : 1.0f;

  constexpr float pi = std::numbers::pi_v<float>;
  const float factor = 2.0f * pi / static_cast<float>(L);

  auto [unit_cell, sublattice] = lattice.from_orbital(op.orbital());

  for (size_t k = 0; k < L; ++k) {
    std::complex<float> coefficient(
        0.0f, -type_sign * factor * static_cast<float>(k * unit_cell));
    coefficient = std::exp(coefficient) / std::sqrt(static_cast<float>(L));

    Operator transformed_op(op.type(), op.spin(),
                            lattice.to_orbital(k, sublattice));
    result += Term(coefficient, {transformed_op});
  }

  return result;
}

static void print_hamiltonian(const Expression& hamiltonian,
                              float epsilon = 3e-4f) {
  for (const auto& [ops, coeff] : hamiltonian.terms()) {
    if (std::abs(coeff) > epsilon) {
      std::cout << "  " << Term(coeff, ops).to_string() << std::endl;
    }
  }
  std::cout << std::endl;
}

static Expression transform_operator_to_band_basis(
    const Operator& op, const arma::cx_fmat& eigenvectors, const Basis& basis) {
  Expression result;
  Operator needle(Operator::Type::Creation, op.spin(), op.orbital());
  size_t i = static_cast<size_t>(basis.index_of({needle}));
  for (size_t k = 0; k < basis.size(); ++k) {
    std::complex<float> coeff = (op.type() == Operator::Type::Annihilation)
                                    ? eigenvectors(i, k)
                                    : std::conj(eigenvectors(i, k));
    // Get spin and orbital indices from basis index k
    Operator::Spin spin = basis.at(k)[0].spin();
    uint8_t orbital = basis.at(k)[0].orbital();
    Term new_term(coeff, {Operator(op.type(), spin, orbital)});
    result += new_term;
  }
  return result;
}

int main() {
  const size_t L = 10;  // 10 unit cells
  float t1 = 1.0f, t2 = 0.5f, delta = 0.1f;

  StaticIndex<L, 2> index;
  SSHModel ssh_model(t1, t2, delta, index);

  std::cout << "SSH Hamiltonian in real space:" << std::endl;
  print_hamiltonian(ssh_model.hamiltonian());

  Expression momentum_hamiltonian =
      transform_expression(fourier_transform_operator<L>,
                           ssh_model.hamiltonian(), ssh_model.index());

  std::cout << "SSH Hamiltonian in momentum space:" << std::endl;
  print_hamiltonian(momentum_hamiltonian);

  Basis basis(2 * L, 1);  // Single-particle basis

  std::cout << "Basis:" << std::endl;
  for (const auto& elm : basis) {
    std::cout << Term(elm).to_string() << std::endl;
  }

  auto H_matrix =
      compute_matrix_elements<arma::cx_fmat>(basis, momentum_hamiltonian);

  arma::fvec eigenvalues;
  arma::cx_fmat eigenvectors;
  arma::eig_sym(eigenvalues, eigenvectors, H_matrix);

  std::cout << "Eigenvalues:" << std::endl;
  for (size_t i = 0; i < eigenvalues.n_elem; ++i) {
    std::cout << basis.at(i).to_string() << "  " << eigenvalues(i) << std::endl;
  }

  Expression diagonal_hamiltonian =
      transform_expression(transform_operator_to_band_basis,
                           momentum_hamiltonian, eigenvectors, basis);

  std::cout << "SSH Hamiltonian in diagonal form:" << std::endl;
  print_hamiltonian(diagonal_hamiltonian);

  return 0;
}
