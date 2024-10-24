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
class TriangularHeisenberg {
 public:
  using Index = StaticIndex<Lx, Ly>;  // 2D triangular lattice

  explicit TriangularHeisenberg(float J) : m_J(J), m_index() {
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

    // Add exchange interactions for triangular lattice
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        uint8_t site = m_index.to_orbital(x, y);

        // Horizontal coupling
        uint8_t right = m_index.to_orbital((x + 1) % Lx, y);
        m_hamiltonian += m_J * Expression::Spin::dot_product(site, right);

        // Vertical coupling
        uint8_t up = m_index.to_orbital(x, (y + 1) % Ly);
        m_hamiltonian += m_J * Expression::Spin::dot_product(site, up);

        // Diagonal coupling (for triangular lattice)
        uint8_t diag = m_index.to_orbital((x + 1) % Lx, (y + 1) % Ly);
        m_hamiltonian += m_J * Expression::Spin::dot_product(site, diag);
      }
    }
  }
};

// Helper function to calculate chirality (scalar spin chirality)
static std::complex<float> compute_chirality(const arma::cx_fvec& state,
                                             const Basis& basis, uint8_t site1,
                                             uint8_t site2, uint8_t site3) {
  // χ = Si · (Sj × Sk)
  Expression chirality_op =
      Expression::Spin::spin_x(site1) *
          (Expression::Spin::spin_y(site2) * Expression::Spin::spin_z(site3) -
           Expression::Spin::spin_z(site2) * Expression::Spin::spin_y(site3)) +
      Expression::Spin::spin_y(site1) *
          (Expression::Spin::spin_z(site2) * Expression::Spin::spin_x(site3) -
           Expression::Spin::spin_x(site2) * Expression::Spin::spin_z(site3)) +
      Expression::Spin::spin_z(site1) *
          (Expression::Spin::spin_x(site2) * Expression::Spin::spin_y(site3) -
           Expression::Spin::spin_y(site2) * Expression::Spin::spin_x(site3));

  auto chirality_matrix =
      compute_matrix_elements<arma::sp_cx_fmat>(basis, chirality_op);
  return arma::cdot(state, chirality_matrix * state);
}

static float compute_spin_correlation(const arma::cx_fvec& state,
                                      const Basis& basis, uint8_t site1,
                                      uint8_t site2) {
  auto corr_op = Expression::Spin::dot_product(site1, site2);
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
void analyze_ground_state(const arma::cx_fvec& ground_state, const Basis& basis,
                          const TriangularHeisenberg<Lx, Ly>& model) {
  const size_t sites = Lx * Ly;

  // Compute total S^z
  float total_sz = compute_total_sz(ground_state, basis, sites);
  std::cout << "Total S^z: " << total_sz << std::endl;

  // Compute average chirality for elementary triangles
  std::complex<float> avg_chirality = 0.0f;
  int triangle_count = 0;
  for (size_t x = 0; x < Lx; ++x) {
    for (size_t y = 0; y < Ly; ++y) {
      uint8_t site1 = model.lattice().to_orbital(x, y);
      uint8_t site2 = model.lattice().to_orbital((x + 1) % Lx, y);
      uint8_t site3 = model.lattice().to_orbital((x + 1) % Lx, (y + 1) % Ly);

      avg_chirality +=
          compute_chirality(ground_state, basis, site1, site2, site3);
      triangle_count++;
    }
  }
  avg_chirality /= static_cast<float>(triangle_count);

  std::cout << "\nAverage scalar chirality: " << avg_chirality << std::endl;

  std::cout << "\nSpin correlations:" << std::endl;
  std::cout << "Distance\tCorrelation" << std::endl;
  std::cout << "---------\t-----------" << std::endl;

  // Compute correlations along different directions
  for (size_t d = 1; d <= std::min(Lx, Ly) / 2; ++d) {
    float avg_correlation = 0.0f;
    int pair_count = 0;

    // Horizontal correlations
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        uint8_t site1 = model.lattice().to_orbital(x, y);
        uint8_t site2 = model.lattice().to_orbital((x + d) % Lx, y);
        avg_correlation +=
            compute_spin_correlation(ground_state, basis, site1, site2);
        pair_count++;
      }
    }

    // Vertical correlations
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        uint8_t site1 = model.lattice().to_orbital(x, y);
        uint8_t site2 = model.lattice().to_orbital(x, (y + d) % Ly);
        avg_correlation +=
            compute_spin_correlation(ground_state, basis, site1, site2);
        pair_count++;
      }
    }

    // Diagonal correlations
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        uint8_t site1 = model.lattice().to_orbital(x, y);
        uint8_t site2 = model.lattice().to_orbital((x + d) % Lx, (y + d) % Ly);
        avg_correlation +=
            compute_spin_correlation(ground_state, basis, site1, site2);
        pair_count++;
      }
    }

    avg_correlation /= static_cast<float>(pair_count);
    std::cout << d << "\t\t" << avg_correlation << std::endl;
  }
}

struct FrustrationSignatures {
  float energy_per_site;
  float energy_variance;
  float avg_local_magnetization;
  float structure_factor_120;
  std::complex<float> avg_chirality;
  float spin_stiffness;
};

static float compute_energy_variance(const arma::cx_fvec& state,
                                     const Basis& basis,
                                     const Expression& hamiltonian) {
  auto H_matrix = compute_matrix_elements<arma::sp_cx_fmat>(basis, hamiltonian);
  auto H2_matrix = H_matrix * H_matrix;

  float E = std::real(arma::cdot(state, H_matrix * state));
  float E2 = std::real(arma::cdot(state, H2_matrix * state));
  return E2 - E * E;
}

static float compute_local_magnetization(const arma::cx_fvec& state,
                                         const Basis& basis, uint8_t site) {
  Expression S2_local =
      Expression::Spin::spin_x(site) * Expression::Spin::spin_x(site) +
      Expression::Spin::spin_y(site) * Expression::Spin::spin_y(site) +
      Expression::Spin::spin_z(site) * Expression::Spin::spin_z(site);

  auto S2_matrix = compute_matrix_elements<arma::sp_cx_fmat>(basis, S2_local);
  return std::sqrt(std::real(arma::cdot(state, S2_matrix * state)));
}

template <size_t Lx, size_t Ly>
static float compute_structure_factor_120(const arma::cx_fvec& state,
                                          const Basis& basis,
                                          const StaticIndex<Lx, Ly>& index) {
  // Q = (4π/3, 0) for 120° order
  const float Qx = 4.0f * std::numbers::pi_v<float> / 3.0f;
  const float Qy = 0.0f;

  Expression structure_factor;
  for (size_t x1 = 0; x1 < Lx; ++x1) {
    for (size_t y1 = 0; y1 < Ly; ++y1) {
      for (size_t x2 = 0; x2 < Lx; ++x2) {
        for (size_t y2 = 0; y2 < Ly; ++y2) {
          uint8_t site1 = index.to_orbital(x1, y1);
          uint8_t site2 = index.to_orbital(x2, y2);

          float phase = Qx * (x1 - x2) + Qy * (y1 - y2);
          std::complex<float> factor(std::cos(phase), std::sin(phase));

          structure_factor +=
              factor * Expression::Spin::dot_product(site1, site2);
        }
      }
    }
  }

  auto S_matrix =
      compute_matrix_elements<arma::sp_cx_fmat>(basis, structure_factor);
  return std::real(arma::cdot(state, S_matrix * state)) / (Lx * Ly);
}

template <size_t Lx, size_t Ly>
static float compute_spin_stiffness(const arma::cx_fvec& state,
                                    const Basis& basis,
                                    const TriangularHeisenberg<Lx, Ly>& model,
                                    float twist_angle) {
  // Create twisted Hamiltonian
  TriangularHeisenberg<Lx, Ly> twisted_model(model.J());
  Expression twisted_H = model.hamiltonian();

  // Add twist along x-direction
  for (size_t y = 0; y < Ly; ++y) {
    for (size_t x = 0; x < Lx; ++x) {
      uint8_t site1 = model.lattice().to_orbital(x, y);
      uint8_t site2 = model.lattice().to_orbital((x + 1) % Lx, y);

      float phase = twist_angle * x / Lx;
      Expression rotation =
          std::cos(phase) * Expression::Spin::dot_product(site1, site2) +
          std::sin(phase) * (Expression::Spin::spin_x(site1) *
                                 Expression::Spin::spin_y(site2) -
                             Expression::Spin::spin_y(site1) *
                                 Expression::Spin::spin_x(site2));

      twisted_H +=
          model.J() * (rotation - Expression::Spin::dot_product(site1, site2));
    }
  }

  auto H_matrix = compute_matrix_elements<arma::sp_cx_fmat>(basis, twisted_H);
  float E_twisted = std::real(arma::cdot(state, H_matrix * state));
  float E_0 =
      std::real(arma::cdot(state, compute_matrix_elements<arma::sp_cx_fmat>(
                                      basis, model.hamiltonian()) *
                                      state));

  return (E_twisted - E_0) / (twist_angle * twist_angle * Lx);
}

template <size_t Lx, size_t Ly>
FrustrationSignatures analyze_frustration(
    const arma::cx_fvec& ground_state, const Basis& basis,
    const TriangularHeisenberg<Lx, Ly>& model) {
  FrustrationSignatures signatures;
  const size_t sites = Lx * Ly;

  // 1. Energy per site and its variance
  auto H_matrix =
      compute_matrix_elements<arma::sp_cx_fmat>(basis, model.hamiltonian());
  signatures.energy_per_site =
      std::real(arma::cdot(ground_state, H_matrix * ground_state)) / sites;
  signatures.energy_variance =
      compute_energy_variance(ground_state, basis, model.hamiltonian()) /
      (sites * sites);

  // 2. Average local magnetization
  float total_local_mag = 0.0f;
  for (size_t i = 0; i < sites; ++i) {
    total_local_mag += compute_local_magnetization(ground_state, basis, i);
  }
  signatures.avg_local_magnetization = total_local_mag / sites;

  // 3. Structure factor for 120° order
  signatures.structure_factor_120 =
      compute_structure_factor_120(ground_state, basis, model.lattice());

  // 4. Average chirality
  std::complex<float> total_chirality = 0.0f;
  int triangle_count = 0;
  for (size_t x = 0; x < Lx; ++x) {
    for (size_t y = 0; y < Ly; ++y) {
      uint8_t site1 = model.lattice().to_orbital(x, y);
      uint8_t site2 = model.lattice().to_orbital((x + 1) % Lx, y);
      uint8_t site3 = model.lattice().to_orbital((x + 1) % Lx, (y + 1) % Ly);

      total_chirality +=
          compute_chirality(ground_state, basis, site1, site2, site3);
      triangle_count++;
    }
  }
  signatures.avg_chirality =
      total_chirality / static_cast<float>(triangle_count);

  // 5. Spin stiffness
  const float small_twist = 0.1f;
  signatures.spin_stiffness =
      compute_spin_stiffness(ground_state, basis, model, small_twist);

  return signatures;
}

static void print_frustration_analysis(const FrustrationSignatures& sigs) {
  std::cout << "\nFrustration Analysis Results:" << std::endl;
  std::cout << "=============================" << std::endl;

  std::cout << std::fixed << std::setprecision(6);

  std::cout << "1. Ground State Energy per site: " << sigs.energy_per_site
            << std::endl;
  std::cout << "   Energy variance per site: " << sigs.energy_variance
            << std::endl;
  std::cout << "   - Large variance indicates quantum fluctuations due to "
               "frustration\n"
            << std::endl;

  std::cout << "2. Average local magnetization: "
            << sigs.avg_local_magnetization << std::endl;
  std::cout << "   - Reduction from S=1/2 (0.5) indicates quantum disorder\n"
            << std::endl;

  std::cout << "3. Structure factor for 120° order: "
            << sigs.structure_factor_120 << std::endl;
  std::cout << "   - Non-zero value indicates tendency toward 120° ordering\n"
            << std::endl;

  std::cout << "4. Average scalar chirality: " << sigs.avg_chirality
            << std::endl;
  std::cout
      << "   - Non-zero imaginary part indicates non-coplanar spin structure\n"
      << std::endl;

  std::cout << "5. Spin stiffness: " << sigs.spin_stiffness << std::endl;
  std::cout
      << "   - Reduced stiffness indicates enhanced quantum fluctuations\n"
      << std::endl;
}

static void print_state_components(const arma::cx_fvec& state_vector,
                                   const Basis& basis) {
  std::cout << "\nState components:" << std::endl;
  for (size_t i = 0; i < basis.size(); ++i) {
    if (std::abs(state_vector(i)) > 3e-4f) {
      Term term(state_vector(i), basis.at(i));
      std::cout << term.to_string() << std::endl;
    }
  }
}

int main() {
  constexpr size_t Lx = 3;  // System size in x direction
  constexpr size_t Ly = 3;  // System size in y direction
  const size_t sites = Lx * Ly;
  const size_t particles = sites;  // Half-filling

  std::cout << "Triangular Lattice Heisenberg Model Analysis" << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << "Lattice size: " << Lx << " x " << Ly << std::endl;
  std::cout << "Number of sites: " << sites << std::endl;
  std::cout << "Number of particles: " << particles << std::endl;

  // Analyze antiferromagnetic case (J > 0), which shows frustration
  float J = 1.0f;
  std::cout << "\nAnalyzing antiferromagnetic coupling J = " << J << std::endl;

  // Construct the model
  TriangularHeisenberg<Lx, Ly> model(J);

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
  analyze_ground_state(ground_state, basis, model);

  auto signatures = analyze_frustration(ground_state, basis, model);
  print_frustration_analysis(signatures);
  return 0;
}
