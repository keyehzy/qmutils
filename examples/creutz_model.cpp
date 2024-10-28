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

template <size_t L>
class CreutzLadderModel {
 public:
  using Index = StaticIndex<L, 2>;  // L unit cells, 2 sites per cell

  CreutzLadderModel(float J, float theta, float U)
      : m_J(J), m_theta(theta), m_U(U), m_index() {
    construct_hamiltonian();
  }

  const Expression& hamiltonian() const { return m_hamiltonian; }
  const Index& index() const { return m_index; }

 private:
  float m_J, m_theta, m_U;
  Index m_index;
  Expression m_hamiltonian;

  void construct_hamiltonian() {
    auto s = Operator::Spin::Up;

    for (size_t i = 0; i < L; ++i) {
      uint8_t A_site = m_index.to_orbital(i, 0);
      uint8_t B_site = m_index.to_orbital(i, 1);

      uint8_t next_A_site = m_index.to_orbital((i + 1) % L, 0);
      uint8_t next_B_site = m_index.to_orbital((i + 1) % L, 1);

      m_hamiltonian += m_J * Expression::hopping(A_site, next_B_site, s);
      m_hamiltonian += m_J * Expression::hopping(B_site, next_A_site, s);

      std::complex<float> phase(std::cos(m_theta), std::sin(m_theta));

      Term A_A = m_J * phase * Term::one_body(s, A_site, s, next_A_site);
      m_hamiltonian += A_A;
      m_hamiltonian += A_A.adjoint();

      Term B_B =
          m_J * std::conj(phase) * Term::one_body(s, B_site, s, next_B_site);
      m_hamiltonian += B_B;
      m_hamiltonian += B_B.adjoint();
    }

    for (size_t i = 0; i < L; ++i) {
      uint8_t A_site = m_index.to_orbital(i, 0);
      uint8_t B_site = m_index.to_orbital(i, 1);

      m_hamiltonian +=
          0.5f * m_U *
          Term({Operator::creation(s, A_site), Operator::creation(s, A_site),
                Operator::annihilation(s, A_site),
                Operator::annihilation(s, A_site)});

      m_hamiltonian +=
          0.5f * m_U *
          Term({Operator::creation(s, B_site), Operator::creation(s, B_site),
                Operator::annihilation(s, B_site),
                Operator::annihilation(s, B_site)});
    }
  }
};

static void print_hamiltonian(const Expression& hamiltonian,
                              float epsilon = 3e-4f) {
  for (const auto& [ops, coeff] : hamiltonian.terms()) {
    if (std::abs(coeff) > epsilon) {
      std::cout << "  " << Term(coeff, ops).to_string() << std::endl;
    }
  }
  std::cout << std::endl;
}

template <size_t L>
static void print_state(const arma::cx_vec& eigvec,
                        const typename CreutzLadderModel<L>::Index& index,
                        const Basis& basis) {
  for (size_t i = 0; i < basis.size(); ++i) {
    // std::cout << Term(basis.at(i)).to_string() << ": " << std::abs(eigvec(i))
    //           << std::endl;
    Basis::operators_type elm = basis.at(i);
    if (elm.empty()) {
      std::cout << std::abs(eigvec(i)) << std::endl;
    } else {
      auto [cell1, j1] =
          index.from_orbital(basis.at(i).operators().front().orbital());
      auto [cell2, j2] =
          index.from_orbital(basis.at(i).operators().back().orbital());
      std::cout << "[" << cell1 << "," << j1 << "] " << "[" << cell2 << ","
                << j2 << "] " << std::abs(eigvec(i)) << std::endl;
    }
  }
  std::cout << "\n\n";
}

std::vector<std::pair<float, float>> calculate_dos(
    const arma::vec& eigenvalues, float sigma = 0.1f, size_t num_points = 1000,
    float padding_factor = 0.1f) {
  // Determine energy range
  float E_min = eigenvalues.min();
  float E_max = eigenvalues.max();
  float padding = (E_max - E_min) * padding_factor;
  E_min -= padding;
  E_max += padding;

  float dE = (E_max - E_min) / static_cast<float>(num_points - 1);
  std::vector<std::pair<float, float>> dos(num_points);

  // Calculate DOS using Gaussian broadening
  const float normalization =
      1.0f / (sigma * std::sqrt(2.0f * std::numbers::pi_v<float>));

#pragma omp parallel for
  for (size_t i = 0; i < num_points; ++i) {
    float E = E_min + i * dE;
    float rho = 0.0f;

    for (size_t j = 0; j < eigenvalues.n_elem; ++j) {
      float delta_E = (E - eigenvalues(j)) / sigma;
      rho += std::exp(-0.5f * delta_E * delta_E);
    }

    dos[i] = {E, rho * normalization};
  }

  return dos;
}

std::vector<std::pair<float, float>> calculate_integrated_dos(
    const std::vector<std::pair<float, float>>& dos) {
  std::vector<std::pair<float, float>> integrated_dos(dos.size());
  float integral = 0.0f;
  float dE = dos[1].first - dos[0].first;

  for (size_t i = 0; i < dos.size(); ++i) {
    integral += dos[i].second * dE;
    integrated_dos[i] = {dos[i].first, integral};
  }

  return integrated_dos;
}

struct ModelParams {
  const size_t L;
  float J;
  float theta;
  float U;
};

void save_dos_data(const arma::vec& eigenvalues, const ModelParams& params,
                   const Basis& basis, const std::string& filename) {
  auto dos = calculate_dos(eigenvalues, 0.1f * params.J);
  auto cumm_dos = calculate_integrated_dos(dos);

  std::ofstream outfile(filename);
  outfile << "# Energy DOS CummulativeDOS\n";

  for (size_t i = 0; i < dos.size(); ++i) {
    outfile << std::setprecision(8) << std::scientific << dos[i].first << " "
            << dos[i].second / static_cast<float>(basis.size()) << " "
            << cumm_dos[i].second / static_cast<float>(basis.size()) << "\n";
  }
}

int main() {
  const size_t L = 20;

  const ModelParams params{
      L,                                 // L
      1.0f,                              // J
      0.5f * std::numbers::pi_v<float>,  // theta
      2.0f,                              // U
  };

  CreutzLadderModel<L> model(params.J, params.theta, params.U);
  Basis basis(2 * params.L, 2);  // Two particle basis

  std::cout << "Basis with: " << basis.size() << " elements." << std::endl;
  for (const auto& elm : basis) {
    std::cout << elm.to_string() << std::endl;
  }

  // std::cout << "Hamiltonian in real space:" << std::endl;
  // print_hamiltonian(model.hamiltonian());

  auto H_matrix =
      compute_matrix_elements<arma::cx_mat>(basis, model.hamiltonian());

  // std::cout << H_matrix.n_rows << " " << H_matrix.n_cols << std::endl;

  arma::vec eigenvalues;
  arma::cx_mat eigenvectors;
  arma::eig_sym(eigenvalues, eigenvectors, H_matrix);

  std::cout << "Eigenvalues:" << std::endl;
  std::cout << eigenvalues << std::endl;

  save_dos_data(eigenvalues, params, basis, "dos_data.dat");

  // std::cout << eigenvalues(basis.size() / 2) << std::endl;
  // print_state<L>(eigenvectors.col(0), model.index(), basis);
  return 0;
}
