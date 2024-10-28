#include "qmutils/basis.h"

#include <unordered_map>

namespace qmutils {

void Basis::generate_basis() {
  operators_type current;
  current.reserve(m_particles);
  generate_combinations(current, 0, 0, m_particles);
}

float Basis::calculate_normalization_factor(const operators_type& ops) const {
  std::unordered_map<Operator, size_t> state_counts;
  for (const auto& op : ops) {
    state_counts[op]++;
  }

  float normalization = 1.0f;
  for (const auto& [op, count] : state_counts) {
    for (size_t i = 2; i <= count; ++i) {
      normalization *= std::sqrt(static_cast<float>(i));
    }
  }

  return 1.0f / normalization;
}

void Basis::generate_combinations(operators_type& current, size_t first_orbital,
                                  size_t depth, size_t max_depth) {
  if (depth == max_depth) {
    operators_type current_copy(current);
    std::sort(current_copy.begin(), current_copy.end());
#ifdef USE_BOSON
    m_basis_states.emplace_back(calculate_normalization_factor(current_copy),
                                current_copy);
#else
    m_basis_states.emplace_back(1.0f, current_copy);
#endif
    return;
  }

  for (size_t i = first_orbital; i < m_orbitals; i++) {
#ifdef USE_BOSON
    for (int spin_index = 0; spin_index < 1; ++spin_index) {
      Operator::Spin spin = static_cast<Operator::Spin>(spin_index);
      if (current.empty() || current.back().orbital() <= i) {
#else
    for (int spin_index = 0; spin_index < 2; ++spin_index) {
      Operator::Spin spin = static_cast<Operator::Spin>(spin_index);
      if (current.empty() || current.back().orbital() < i ||
          ((current.back().orbital() == i && spin > current.back().spin()))) {
#endif
        current.push_back(Operator::creation(spin, i));
        generate_combinations(current, i, depth + 1, max_depth);
        current.pop_back();
      }
    }
  }
}
}  // namespace qmutils
