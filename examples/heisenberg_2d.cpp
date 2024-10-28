#include <armadillo>
#include <iostream>
#include <vector>

#include "qmutils/basis.h"
#include "qmutils/expression.h"
#include "qmutils/index.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/normal_order.h"
#include "qmutils/operator.h"

using namespace qmutils;

template <size_t Lx, size_t Ly>
class Heisenberg2D {
 public:
  using Index = StaticIndex<Lx, Ly>;  // 2D square lattice

  explicit Heisenberg2D(float J) : m_J(J), m_index() {
    construct_hamiltonian();
  }

  const Expression& hamiltonian() const { return m_hamiltonian; }
  const Index& lattice() const { return m_index; }
  float J() const { return m_J; }

 private:
  float m_J;
  Index m_index;
  Expression m_hamiltonian;

  void construct_hamiltonian() {
    // Add small magnetic field term to lift degeneracy
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        uint8_t site = m_index.to_orbital(x, y);
        m_hamiltonian += -3e-3f * Expression::Spin::spin_z(site);
      }
    }

    // Add exchange interactions
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        uint8_t site = m_index.to_orbital(x, y);

        // Horizontal coupling
        uint8_t right = m_index.to_orbital((x + 1) % Lx, y);
        m_hamiltonian += m_J * Expression::Spin::dot_product(site, right);

        // Vertical coupling
        uint8_t up = m_index.to_orbital(x, (y + 1) % Ly);
        m_hamiltonian += m_J * Expression::Spin::dot_product(site, up);
      }
    }
  }
};

static float compute_spin_correlation(const arma::cx_fvec& state,
                                      const Basis& basis, uint8_t site1,
                                      uint8_t site2) {
  auto corr_op =
      Expression::Spin::spin_z(site1) * Expression::Spin::spin_z(site2);
  auto corr_matrix = compute_matrix_elements<arma::sp_cx_fmat>(basis, corr_op);
  return std::real(arma::cdot(state, corr_matrix * state));
}

static float compute_total_sz(const arma::cx_fvec& state, const Basis& basis,
                              size_t sites) {
  Expression total_sz;
  for (uint8_t i = 0; i < sites; ++i) {
    total_sz += Expression::Spin::spin_z(i);
  }
  auto sz_matrix = compute_matrix_elements<arma::sp_cx_fmat>(basis, total_sz);
  return std::real(arma::cdot(state, sz_matrix * state));
}

template <size_t Lx, size_t Ly>
void analyze_ground_state(const arma::cx_fvec& ground_state,
                          const Basis& basis) {
  const size_t sites = Lx * Ly;

  // Compute total S^z
  float total_sz = compute_total_sz(ground_state, basis, sites);
  std::cout << "Total S^z: " << total_sz << std::endl;

  std::cout << "\nSpin correlations:" << std::endl;
  std::cout << "Distance\tCorrelation" << std::endl;
  std::cout << "---------\t-----------" << std::endl;

  // Compute correlations along x direction
  for (size_t dx = 1; dx <= Lx / 2; ++dx) {
    float avg_correlation = 0.0f;
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        uint8_t site1 = static_cast<uint8_t>(y * Lx + x);
        uint8_t site2 = static_cast<uint8_t>(y * Lx + ((x + dx) % Lx));
        avg_correlation +=
            compute_spin_correlation(ground_state, basis, site1, site2);
      }
    }
    avg_correlation /= static_cast<float>(Lx * Ly);
    std::cout << dx << "\t\t" << avg_correlation << std::endl;
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
  constexpr size_t Lx = 4;  // System size in x direction
  constexpr size_t Ly = 2;  // System size in y direction
  const size_t sites = Lx * Ly;
  const size_t particles = sites;  // Half-filling

  std::cout << "2D Heisenberg Model Analysis" << std::endl;
  std::cout << "===========================" << std::endl;
  std::cout << "Lattice size: " << Lx << " x " << Ly << std::endl;
  std::cout << "Number of sites: " << sites << std::endl;
  std::cout << "Number of particles: " << particles << std::endl;

  // Analyze both ferromagnetic and antiferromagnetic cases
  std::vector<float> J_values = {-1.0f, 1.0f};

  for (float J : J_values) {
    std::cout << "\nAnalyzing coupling J = " << J << std::endl;
    std::cout << (J < 0 ? "Ferromagnetic case:" : "Antiferromagnetic case:")
              << std::endl;

    // Construct the model
    Heisenberg2D<Lx, Ly> model(J);

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
    analyze_ground_state<Lx, Ly>(ground_state, basis);
  }

  return 0;
}
