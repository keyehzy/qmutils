#include <armadillo>
#include <iostream>
#include <vector>

#include "qmutils/basis.h"
#include "qmutils/expression.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/normal_order.h"
#include "qmutils/operator.h"

using namespace qmutils;

class Heisenberg1D {
 public:
  explicit Heisenberg1D(size_t sites, float J) : m_sites(sites), m_J(J) {}

  size_t sites() const { return m_sites; }
  float J() const { return m_J; }

  Expression hamiltonian() const {
    Expression result;
    for (uint8_t i = 0; i < m_sites; i++) {
      result += -3e-3f * Expression::Spin::spin_z(i);
    }

    for (uint8_t i = 0; i < m_sites; ++i) {
      result += m_J * Expression::Spin::dot_product(i, (i + 1) % m_sites);
    }
    return result;
  }

 private:
  size_t m_sites;
  float m_J;
};

static float compute_spin_correlation(const arma::cx_fvec& state,
                                      const Basis& basis, uint8_t site1,
                                      uint8_t site2) {
  auto corr_op =
      Expression::Spin::spin_z(site1) * Expression::Spin::spin_z(site2);
  auto corr_matrix = compute_matrix_elements<arma::cx_fmat>(basis, corr_op);
  return std::real(arma::cdot(state, corr_matrix * state));
}

static float compute_total_sz(const arma::cx_fvec& state, const Basis& basis,
                              size_t sites) {
  Expression total_sz;
  for (uint8_t i = 0; i < sites; ++i) {
    total_sz += Expression::Spin::spin_z(i);
  }
  auto sz_matrix = compute_matrix_elements<arma::cx_fmat>(basis, total_sz);
  return std::real(arma::cdot(state, sz_matrix * state));
}

static void analyze_ground_state(const arma::cx_fvec& ground_state,
                                 const Basis& basis, size_t sites) {
  // Compute total S^z
  float total_sz = compute_total_sz(ground_state, basis, sites);
  std::cout << "Total S^z: " << total_sz << std::endl;

  std::cout << "\nSpin correlations:" << std::endl;
  std::cout << "Distance\tCorrelation" << std::endl;
  std::cout << "---------\t-----------" << std::endl;

  // Compute correlations for different distances
  for (uint8_t d = 1; d <= sites / 2; ++d) {
    float avg_correlation = 0.0f;
    for (uint8_t i = 0; i < sites; ++i) {
      avg_correlation +=
          compute_spin_correlation(ground_state, basis, i, (i + d) % sites);
    }
    avg_correlation /= static_cast<float>(sites);

    std::cout << d << "\t\t" << avg_correlation << std::endl;
  }
}

static void print_state_components(const arma::cx_fvec& state_vector,
                                   const Basis& basis) {
  std::cout << "\nState components:" << std::endl;
  for (size_t i = 0; i < basis.size(); ++i) {
    if (std::abs(state_vector(i)) > 3e-4f) {
      Term term(state_vector(i), basis.at(i).operators());
      std::cout << term.to_string() << std::endl;
    }
  }
}

int main() {
  const size_t sites = 6;      // Number of sites in the chain
  const size_t particles = 6;  // Half-filling

  std::cout << "1D Heisenberg Model Analysis" << std::endl;
  std::cout << "===========================" << std::endl;
  std::cout << "Number of sites: " << sites << std::endl;
  std::cout << "Number of particles: " << particles << std::endl;

  // Analyze both ferromagnetic and antiferromagnetic cases
  std::vector<float> J_values = {-1.0f, 1.0f};

  for (float J : J_values) {
    std::cout << "\nAnalyzing coupling J = " << J << std::endl;
    std::cout << (J < 0 ? "Ferromagnetic case:" : "Antiferromagnetic case:")
              << std::endl;

    // Construct the model
    Heisenberg1D model(sites, J);

    // Construct the basis
    Basis basis(sites, particles);

    // Compute Hamiltonian matrix
    auto H_matrix =
        compute_matrix_elements<arma::sp_cx_fmat>(basis, model.hamiltonian());

    // Find eigenvalues and eigenvectors
    arma::cx_fvec eigenvalues;
    arma::cx_fmat eigenvectors;
    arma::eigs_gen(eigenvalues, eigenvectors, H_matrix, 1, "sr");

    // Ground state is the eigenvector corresponding to lowest eigenvalue
    arma::cx_fvec ground_state = eigenvectors.col(0);
    float ground_energy = eigenvalues(0).real();

    std::cout << "Ground state energy per site: "
              << ground_energy / static_cast<float>(sites) << std::endl;

    // Analyze the ground state
    print_state_components(ground_state, basis);
    analyze_ground_state(ground_state, basis, sites);
  }

  return 0;
}
