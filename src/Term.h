#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "Operator.h"

namespace qmutils {

class Term {
 public:
  using coefficient_type = std::complex<float>;
  using container_type = std::vector<Operator>;

  Term() = default;

  Term(Operator op) : m_coefficient(1.0f), m_operators({op}) {}

  Term(const coefficient_type& coeff) : m_coefficient(coeff) {}

  Term(const coefficient_type& coeff, const container_type& operators)
      : m_coefficient(coeff), m_operators(operators) {}

  Term(const coefficient_type& coeff, container_type&& operators)
      : m_coefficient(coeff), m_operators(std::move(operators)) {}

  Term(const Term& other) = default;

  Term(Term&& other) noexcept = default;

  Term& operator=(const Term& other) = default;

  Term& operator=(Term&& other) noexcept = default;

  const container_type& operators() const { return m_operators; }

  const coefficient_type& coefficient() const { return m_coefficient; }

  size_t size() const { return m_operators.size(); }

  Operator operator[](size_t index) const { return m_operators[index]; }

  Operator& operator[](size_t index) { return m_operators[index]; }

  Term& operator*=(const Term& other) {
    m_coefficient *= other.m_coefficient;
    m_operators.reserve(m_operators.size() + other.m_operators.size());
    m_operators.insert(m_operators.end(), other.m_operators.begin(),
                       other.m_operators.end());
    return *this;
  }

  Term& operator*=(Operator op) {
    m_operators.push_back(op);
    return *this;
  }

  Term& operator*=(const coefficient_type& scalar) {
    m_coefficient *= scalar;
    return *this;
  }

  bool operator==(const Term& other) const {
    return m_coefficient == other.m_coefficient &&
           m_operators == other.m_operators;
  }

  bool operator!=(const Term& other) const { return !(*this == other); }

  Term adjoint() const noexcept;

  std::string to_string() const;

  static Term creation(Operator::Spin spin, uint8_t orbital);
  static Term annihilation(Operator::Spin spin, uint8_t orbital);
  static Term number(Operator::Spin spin, uint8_t orbital);
  static Term spin_flip(uint8_t orbital);
  static Term hopping(uint8_t from_orbital, uint8_t to_orbital,
                      Operator::Spin spin);

 private:
  coefficient_type m_coefficient;
  container_type m_operators;
};

inline Term operator*(Term term, const Term& other) {
  term *= other;
  return term;
}

inline Term operator*(Term term, Operator other) {
  term *= other;
  return term;
}

inline Term operator*(Term term, const Term::coefficient_type& scalar) {
  term *= scalar;
  return term;
}

inline Term operator*(const Term::coefficient_type& scalar, Term term) {
  return term * scalar;
}

}  // namespace qmutils

template <>
struct std::hash<qmutils::Term> {
  size_t operator()(const qmutils::Term& term) const {
    size_t hash = std::hash<float>{}(term.coefficient().real()) ^
                  std::hash<float>{}(term.coefficient().imag());
    for (const auto& op : term.operators()) {
      hash_combine(hash, std::hash<qmutils::Operator>{}(op));
    }
    return hash;
  }

 private:
  template <typename T>
  void hash_combine(size_t& seed, const T& value) const {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
};
