#include "qmutils/normal_order.h"

#include <iostream>

namespace qmutils {

Expression NormalOrderer::normal_order(const Term& term) {
  return term.coefficient() * normal_order_iterative(term.operators());
}

Expression NormalOrderer::normal_order(const Expression& expr) {
  Expression result;
  for (const auto& [ops, coeff] : expr.terms()) {
    result += coeff * normal_order_iterative(ops);
  }
  return result;
}

Expression NormalOrderer::normal_order_iterative(const operators_type& ops) {
  Expression result;
  m_queue.emplace(ops, 1.0);

  while (!m_queue.empty()) {
    auto [current, phase] = m_queue.top();
    m_queue.pop();

    if (current.size() < 2) {
      result += Term(phase, current);
      continue;
    }

    auto ops_key = std::hash<operators_type>{}(current);
    Expression cached_result;
    if (m_cache.get(ops_key, cached_result)) {
      m_cache_hits++;
      result += phase * cached_result;
      continue;
    }
    m_cache_misses++;

    bool is_sorted = true;
    Expression partial;

    for (size_t i = 1; i < current.size(); ++i) {
      size_t j = i;
      while (j > 0 && current[j] < current[j - 1]) {
        if (current[j].commutes_with(current[j - 1])) {
          std::swap(current[j], current[j - 1]);
          phase *= -1.0;
          --j;
        } else {
          is_sorted = false;
          operators_type contracted(current);
          contracted.erase(contracted.begin() + j - 1,
                           contracted.begin() + j + 1);
          m_queue.emplace(std::move(contracted), phase);

          operators_type swapped(current);
          std::swap(swapped[j - 1], swapped[j]);
          m_queue.emplace(std::move(swapped), -phase);
          break;
        }
      }
      if (!is_sorted) break;
    }

    if (is_sorted) {
      partial = Expression(Term(phase, current));
    }

    m_cache.put(ops_key, partial);
    result += partial;
  }

  return result;
}

Expression NormalOrderer::normal_order_recursive(operators_type ops) {
  if (ops.size() < 2) {
    return Expression(Term(ops));
  }

  auto ops_key = std::hash<operators_type>{}(ops);
  Expression cached_result;
  if (m_cache.get(ops_key, cached_result)) {
    m_cache_hits++;
    return cached_result;
  }
  m_cache_misses++;

  coefficient_type phase = 1.0;
  for (size_t i = 1; i < ops.size(); ++i) {
    size_t j = i;
    while (j > 0 && ops[j] < ops[j - 1]) {
      if (ops[j].commutes_with(ops[j - 1])) {
        std::swap(ops[j], ops[j - 1]);
        phase *= -1.0;
        --j;
      } else {
        Expression result = phase * handle_non_commuting(ops, j - 1);
        m_cache.put(ops_key, result);
        return result;
      }
    }
  }

  Expression result(Term(phase, ops));
  m_cache.put(ops_key, result);
  return result;
}

Expression NormalOrderer::handle_non_commuting(const operators_type& ops,
                                               size_t index) {
  operators_type contracted(ops);
  contracted.erase(contracted.begin() + index, contracted.begin() + index + 2);
  operators_type swapped(ops);
  std::swap(swapped[index], swapped[index + 1]);
  return normal_order_recursive(contracted) - normal_order_recursive(swapped);
}

void NormalOrderer::print_cache_stats() const {
  std::cout << "Total entries: " << m_cache.size() << std::endl;
  std::cout << "Cache hits: " << m_cache_hits << std::endl;
  std::cout << "Cache misses: " << m_cache_misses << std::endl;
  double hit_ratio =
      static_cast<double>(m_cache_hits) / (m_cache_hits + m_cache_misses);
  std::cout << "Hit ratio: " << hit_ratio << std::endl;
}

}  // namespace qmutils
