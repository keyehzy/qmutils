#include <armadillo>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "qmutils/expression.h"
#include "qmutils/functional.h"
#include "qmutils/index.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/operator.h"

using namespace qmutils;

template <typename T, std::size_t D>
struct Vec {
  std::array<T, D> components;

  Vec() : components{} {}

  Vec(std::initializer_list<T> init) {
    if (init.size() != D) {
      throw std::invalid_argument(
          "Initializer list size does not match vector dimension");
    }
    std::copy(init.begin(), init.end(), components.begin());
  }

  T& operator[](std::size_t index) { return components[index]; }

  const T& operator[](std::size_t index) const { return components[index]; }

  Vec operator+(const Vec& other) const {
    Vec result;
    for (std::size_t i = 0; i < D; ++i) {
      result[i] = components[i] + other[i];
    }
    return result;
  }

  Vec operator*(T other) const {
    Vec result;
    for (std::size_t i = 0; i < D; ++i) {
      result[i] = components[i] * other;
    }
    return result;
  }
};

template <size_t Lx, size_t Ly>
class GrapheneModel {
 public:
  using Position = Vec<float, 2>;
  using LatticeMap = std::unordered_map<size_t, Position>;

  GrapheneModel(float t) : m_t(t) {
    construct_lattice();
    construct_hamiltonian();
  }

  const Expression& hamiltonian() const { return m_hamiltonian; }
  const StaticIndex<Lx, Ly, 2>& lattice() const { return m_index; }
  const LatticeMap& positions() const { return m_positions; }

 private:
  float m_t;
  Expression m_hamiltonian;
  StaticIndex<Lx, Ly, 2> m_index;
  LatticeMap m_positions;

  void construct_lattice() {
    const float a = 1.0f;  // lattice constant
    const float sqrt3 = std::sqrt(3.0f);
    const Position a1{(sqrt3 * a) / 2.0f, a / 2.0f};
    const Position a2{(sqrt3 * a) / 2.0f, -a / 2.0f};
    const Position d1{0.0f, a / sqrt3};

    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        // A sublattice
        m_positions[m_index.to_orbital(x, y, 0)] = a1 * x + a2 * y;
        // B sublattice
        m_positions[m_index.to_orbital(x, y, 1)] = a1 * x + a2 * y + d1;
      }
    }
  }

  void construct_hamiltonian() {
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        // Intra-cell hopping
        m_hamiltonian += hopping(x, y, 0, x, y, 1);

        // Inter-cell
        m_hamiltonian += hopping(x, y, 1, (x + 1) % Lx, y, 0);
        m_hamiltonian += hopping(x, y, 1, x, (y + 1) % Ly, 0);
      }
    }
  }

  Expression hopping(size_t x_1, size_t y_1, size_t site_1, size_t x_2,
                     size_t y_2, size_t site_2) {
    uint8_t orbital1 = m_index.to_orbital(x_1, y_1, site_1);
    uint8_t orbital2 = m_index.to_orbital(x_2, y_2, site_2);
    Expression term;
    term += m_t * Expression::hopping(orbital1, orbital2, Operator::Spin::Up);
    term += m_t * Expression::hopping(orbital1, orbital2, Operator::Spin::Down);
    return term;
  }
};

template <size_t Lx, size_t Ly>
Expression fourier_transform_operator(
    const Operator& op, const StaticIndex<Lx, Ly, 2>& lattice,
    const typename GrapheneModel<Lx, Ly>::LatticeMap& positions) {
  Expression result;
  const float type_sign =
      (op.type() == Operator::Type::Annihilation) ? -1.0f : 1.0f;
  constexpr float pi = std::numbers::pi_v<float>;
  const float a = 1.0f;
  const float sqrt3 = std::sqrt(3.0f);
  const float x_factor = 1.0f;  // 2.0f * pi / (Lx * sqrt3 * a);
  const float y_factor = 1.0f;  // 2.0f * pi / (Ly * 3.0f * a);

  auto [x, y, s] = lattice.from_orbital(op.orbital());
  const auto& pos = positions.at(op.orbital());

  for (size_t kx = 0; kx < Lx; ++kx) {
    for (size_t ky = 0; ky < Ly; ++ky) {
      float kx_val = kx * x_factor;
      float ky_val = ky * y_factor;

      std::complex<float> exponent(
          0.0f, type_sign * (kx_val * pos[0] + ky_val * pos[1]));
      std::complex<float> coefficient =
          std::exp(exponent) / std::sqrt(static_cast<float>(Lx * Ly));

      Operator transformed_op(op.type(), op.spin(),
                              lattice.to_orbital(kx, ky, s));
      result += Term(coefficient, {transformed_op});
    }
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

int main() {
  constexpr size_t Lx = 5;
  constexpr size_t Ly = 5;
  const float t = 1.0f;  // hopping strength

  GrapheneModel<Lx, Ly> model(t);

  std::cout << "Constructing momentum space Hamiltonian..." << std::endl;
  Expression momentum_hamiltonian = transform_expression(
      [&](const Operator& op) {
        return fourier_transform_operator<Lx, Ly>(op, model.lattice(),
                                                  model.positions());
      },
      model.hamiltonian());

  std::cout << "Hamiltonian in momentum space:" << std::endl;
  print_hamiltonian(momentum_hamiltonian);

  std::cout << "Setting up basis..." << std::endl;
  Basis basis(model.lattice().size(), 1);  // Single-particle basis

  std::cout << "Computing matrix elements..." << std::endl;
  auto H_matrix =
      compute_matrix_elements<arma::cx_fmat>(basis, momentum_hamiltonian);

  std::cout << "Diagonalizing..." << std::endl;
  arma::fvec eigenvalues;
  arma::cx_fmat eigenvectors;
  arma::eig_sym(eigenvalues, eigenvectors, H_matrix);

  const float a = 1.0f;  // Lattice constant
  const float sqrt3 = std::sqrt(3.0f);
  const float kx_factor =
      1.0f;  // 2.0f * std::numbers::pi_v<float> / (Lx * sqrt3 * a);
  const float ky_factor =
      1.0f;  // 2.0f * std::numbers::pi_v<float> / (Ly * 3.0f * a);

  std::cout << "kx ky n Energy" << std::endl;
  for (size_t i = 0; i < eigenvalues.n_elem; ++i) {
    size_t orbital_index = i % basis.orbitals();
    auto [kx_index, ky_index, n] = model.lattice().from_orbital(orbital_index);

    // Calculate actual kx and ky values
    float kx = kx_index * kx_factor;
    float ky = ky_index * ky_factor;

    std::cout << kx << " " << ky << " " << n << " " << eigenvalues(i)
              << std::endl;
  }
  return 0;
}
