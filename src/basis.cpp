#include "qmutils/basis.h"

namespace qmutils {

void Basis::generate_basis() {
  operators_type current;
  current.reserve(m_particles);
#ifdef USE_BOSON
  for (size_t n = 0; n <= m_particles; ++n) {
    generate_combinations(current, 0, 0, n);
  }
#else
  generate_combinations(current, 0, 0, m_particles);
#endif
}

void Basis::generate_combinations(operators_type& current, size_t first_orbital,
                                  size_t depth, size_t max_depth) {
  if (depth == max_depth) {
    operators_type current_copy(current);
    std::sort(current_copy.begin(), current_copy.end());
    m_index_map.push_back(current_copy);
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
