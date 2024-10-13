#include <benchmark/benchmark.h>

#include <random>

#include "qmutils/term.h"

namespace qmutils {
namespace {

Term::container_type generateLargeOperatorSequence(size_t size) {
  Term::container_type ops;
  ops.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    ops.emplace_back(Operator::Type(i % 2), Operator::Spin(i % 2), i % 64);
  }
  return ops;
}

std::complex<float> randomComplex() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dis(-1.0, 1.0);
  return {dis(gen), dis(gen)};
}

static void BM_ConstructLargeTerm(benchmark::State& state) {
  const size_t largeSize = state.range(0);
  auto largeOps = generateLargeOperatorSequence(largeSize);

  for (auto _ : state) {
    Term largeTerm(randomComplex(), largeOps);
    benchmark::DoNotOptimize(largeTerm);
  }
  state.SetItemsProcessed(state.iterations() * largeSize);
}
BENCHMARK(BM_ConstructLargeTerm)->Range(1 << 10, 1 << 20);

static void BM_MultiplyLargeTerms(benchmark::State& state) {
  const size_t size = state.range(0);
  Term term1(randomComplex(), generateLargeOperatorSequence(size));
  Term term2(randomComplex(), generateLargeOperatorSequence(size));

  for (auto _ : state) {
    Term result = term1 * term2;
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations() * size * 2);
}
BENCHMARK(BM_MultiplyLargeTerms)->Range(1 << 10, 1 << 15);

static void BM_HashLargeTerm(benchmark::State& state) {
  const size_t largeSize = state.range(0);
  Term largeTerm(randomComplex(), generateLargeOperatorSequence(largeSize));
  std::hash<Term> hasher;

  for (auto _ : state) {
    size_t hash = hasher(largeTerm);
    benchmark::DoNotOptimize(hash);
  }
  state.SetItemsProcessed(state.iterations() * largeSize);
}
BENCHMARK(BM_HashLargeTerm)->Range(1 << 10, 1 << 17);

}  // namespace
}  // namespace qmutils
