#pragma once

#include <algorithm>
#include <unordered_set>

#include "qmutils/assert.h"
#include "qmutils/operator.h"
#include "qmutils/term.h"

namespace qmutils {

static constexpr uint64_t qmutils_choose(uint64_t n, uint64_t m) {
  if (m > n) return 0;
  if (m == 0 || m == n) return 1;

  if (m > n - m) m = n - m;

  uint64_t result = 1;
  for (uint64_t i = 0; i < m; ++i) {
    result *= (n - i);
    result /= (i + 1);
  }

  return result;
}

static constexpr uint64_t qmutils_compute_basis_size(size_t orbitals,
                                                     size_t particles) {
#ifdef USE_BOSON
  size_t total_size = 1;
  for (size_t n = 1; n <= particles; ++n) {
    total_size += qmutils_choose(orbitals + n - 1, n);
  }
#else
  size_t total_size = qmutils_choose(2 * orbitals, particles);
#endif
  return total_size;
}

class Basis {
 public:
  using operators_type = Term::container_type;

  Basis(size_t orbitals, size_t particles)
      : m_orbitals(orbitals), m_particles(particles) {
#ifndef USE_BOSON
    QMUTILS_ASSERT(particles <= 2 * orbitals);
#endif
    m_basis_states.reserve(qmutils_choose(2 * orbitals, particles));
    generate_basis();
    std::sort(m_basis_states.begin(), m_basis_states.end(),
              [](const Term& a, const Term& b) {
                return a.operators() < b.operators();
              });
  }

  Basis(const Basis&) = default;
  Basis& operator=(const Basis&) = default;
  Basis(Basis&&) noexcept = default;
  Basis& operator=(Basis&&) noexcept = default;

  ~Basis() = default;

  size_t orbitals() const noexcept { return m_orbitals; }
  size_t particles() const noexcept { return m_particles; }
  size_t size() const noexcept { return m_basis_states.size(); }

  bool operator==(const Basis& other) const {
    return m_orbitals == other.m_orbitals && m_particles == other.m_particles &&
           m_basis_states == other.m_basis_states;
  }

  bool operator!=(const Basis& other) const { return !(*this == other); }

  bool contains(const operators_type& ops) const {
    Term term(ops);
    auto it = std::lower_bound(m_basis_states.begin(), m_basis_states.end(),
                               term, [](const Term& a, const Term& b) {
                                 return a.operators() < b.operators();
                               });
    return it != m_basis_states.end() && it->operators() == ops;
  }

  ptrdiff_t index_of(const operators_type& ops) const {
    QMUTILS_ASSERT(contains(ops));
    Term term(ops);
    auto it = std::lower_bound(m_basis_states.begin(), m_basis_states.end(),
                               term, [](const Term& a, const Term& b) {
                                 return a.operators() < b.operators();
                               });
    return std::distance(m_basis_states.begin(), it);
  }

  void insert(const operators_type& ops) {
    QMUTILS_ASSERT(!contains(ops));
    Term term(ops);
    auto it = std::lower_bound(m_basis_states.begin(), m_basis_states.end(),
                               term, [](const Term& a, const Term& b) {
                                 return a.operators() < b.operators();
                               });
    m_basis_states.insert(it, term);
  }

  auto begin() const noexcept { return m_basis_states.begin(); }
  auto end() const noexcept { return m_basis_states.end(); }

  auto at(size_t i) const noexcept {
    QMUTILS_ASSERT(i < m_basis_states.size());
    return m_basis_states[i];
  }

 private:
  void generate_basis();

  void generate_combinations(operators_type& current, size_t first_orbital,
                             size_t depth, size_t max_depth);

  float calculate_normalization_factor(const operators_type& ops) const;

  size_t m_orbitals;
  size_t m_particles;
  std::vector<Term> m_basis_states;
};

}  // namespace qmutils
