#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "qmutils/expression.h"
#include "qmutils/normal_order.h"

namespace qmutils {
namespace {

Operator random_operator() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> type_dist(0, 1);
  static std::uniform_int_distribution<> spin_dist(0, 1);
  static std::uniform_int_distribution<> orbital_dist(0, 63);

  Operator::Type type = static_cast<Operator::Type>(type_dist(gen));
  Operator::Spin spin = static_cast<Operator::Spin>(spin_dist(gen));
  uint8_t orbital = static_cast<uint8_t>(orbital_dist(gen));

  return Operator(type, spin, orbital);
}

Term create_random_term(size_t n_operators) {
  Term::container_type operators;
  operators.reserve(n_operators);
  for (size_t i = 0; i < n_operators; ++i) {
    operators.push_back(random_operator());
  }
  return Term(Term::coefficient_type(1.0f, 0.0f), operators);
}

static void BM_LRUCacheNormalOrderRandom(benchmark::State& state) {
  const size_t cache_size = state.range(0);
  const size_t n_operators = state.range(1);
  const size_t n_terms = 10000;  // Number of terms to normal order

  std::vector<Term> terms;
  terms.reserve(n_terms);
  for (size_t i = 0; i < n_terms; ++i) {
    terms.push_back(create_random_term(n_operators));
  }

  for (auto _ : state) {
    NormalOrderer orderer(cache_size);
    for (const auto& term : terms) {
      benchmark::DoNotOptimize(orderer.normal_order(term));
    }
  }

  state.SetComplexityN(n_terms * n_operators);
}

Term create_noncommuting_term(size_t n_operators) {
  Term::container_type operators;
  operators.reserve(n_operators);

  for (size_t i = 0; i < n_operators / 2; ++i) {
    operators.push_back(Operator::annihilation(Operator::Spin::Up, 0));
  }

  for (size_t i = 0; i < n_operators / 2; ++i) {
    operators.push_back(Operator::creation(Operator::Spin::Up, 0));
  }

  return Term(Term::coefficient_type(1.0f, 0.0f), std::move(operators));
}

static void BM_LRUCacheNormalOrderNonCommutingTerm(benchmark::State& state) {
  const size_t cache_size = state.range(0);
  const size_t n_operators = state.range(1);
  const size_t n_terms = 10000;  // Number of terms to normal order

  std::vector<Term> terms;
  terms.reserve(n_terms);
  for (size_t i = 0; i < n_terms; ++i) {
    terms.push_back(create_noncommuting_term(n_operators));
  }

  for (auto _ : state) {
    NormalOrderer orderer(cache_size);
    for (const auto& term : terms) {
      benchmark::DoNotOptimize(orderer.normal_order(term));
    }
  }

  state.SetComplexityN(n_terms * n_operators);
}

Term generate_momentum_conserving_term(uint8_t nk) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, nk - 1);

  uint8_t k1 = dist(gen);
  uint8_t k2 = dist(gen);
  uint8_t k3 = dist(gen);
  uint8_t k4 = (k1 + k2 - k3 + nk) % nk;  // Ensure momentum conservation

  return Term(Term::coefficient_type(1.0f, 0.0f),
              {Operator::creation(Operator::Spin::Up, k1),
               Operator::creation(Operator::Spin::Down, k2),
               Operator::annihilation(Operator::Spin::Up, k3),
               Operator::annihilation(Operator::Spin::Down, k4)});
}

static void BM_LRUCacheNormalOrderMomentumPreserving(benchmark::State& state) {
  const size_t cache_size = state.range(0);
  const size_t n_operators = state.range(1);
  const size_t n_terms = 10000;  // Number of terms to normal order

  std::vector<Term> terms;
  terms.reserve(n_terms);
  for (size_t i = 0; i < n_terms; ++i) {
    terms.push_back(generate_momentum_conserving_term(n_operators));
  }

  for (auto _ : state) {
    NormalOrderer orderer(cache_size);
    for (const auto& term : terms) {
      benchmark::DoNotOptimize(orderer.normal_order(term));
    }
  }

  state.SetComplexityN(n_terms * n_operators);
}

BENCHMARK(BM_LRUCacheNormalOrderRandom)
    ->Ranges({{1 << 10, 1 << 30}, {1, 64}})
    ->Complexity();

BENCHMARK(BM_LRUCacheNormalOrderNonCommutingTerm)
    ->Ranges({{1 << 10, 1 << 30}, {1, 64}})
    ->Complexity();

BENCHMARK(BM_LRUCacheNormalOrderMomentumPreserving)
    ->Ranges({{1 << 10, 1 << 30}, {1, 64}})
    ->Complexity();

}  // namespace
}  // namespace qmutils
