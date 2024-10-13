// term_test.cpp
#include "qmutils/term.h"

#include <gtest/gtest.h>

#include <chrono>

namespace qmutils {
namespace {

class TermTest : public ::testing::Test {
 protected:
  Operator op1 = Operator::creation(Operator::Spin::Up, 0);
  Operator op2 = Operator::annihilation(Operator::Spin::Down, 1);
  std::complex<float> coeff{0.5f, -0.5f};
};

TEST_F(TermTest, DefaultConstructor) {
  Term term;
  EXPECT_EQ(term.size(), 0);
  EXPECT_EQ(term.coefficient(), std::complex<float>(0.0f, 0.0f));
}

TEST_F(TermTest, ConstructorWithOperatorsAndCoefficient) {
  Term term(coeff, {op1, op2});
  EXPECT_EQ(term.size(), 2);
  EXPECT_EQ(term.coefficient(), coeff);
  EXPECT_EQ(term[0], op1);
  EXPECT_EQ(term[1], op2);
}

TEST_F(TermTest, MoveConstructor) {
  Term term1(coeff, {op1, op2});
  Term term2(std::move(term1));
  EXPECT_EQ(term2.size(), 2);
  EXPECT_EQ(term2.coefficient(), coeff);
  EXPECT_EQ(term2[0], op1);
  EXPECT_EQ(term2[1], op2);
  // term1 should be in a valid but unspecified state after move
  EXPECT_EQ(term1.size(), 0);
}

TEST_F(TermTest, MultiplyOperator) {
  Term term;
  term *= op1;
  EXPECT_EQ(term.size(), 1);
  EXPECT_EQ(term[0], op1);
  term *= op2;
  EXPECT_EQ(term.size(), 2);
  EXPECT_EQ(term[1], op2);
}

TEST_F(TermTest, GetCoefficient) {
  Term term(42);
  EXPECT_EQ(term.coefficient(), std::complex<float>(42.0f, 0.0f));
}

TEST_F(TermTest, AccessOperators) {
  Term term(coeff, {op1, op2});
  EXPECT_EQ(term.size(), 2);
  EXPECT_EQ(term[0], op1);
  EXPECT_EQ(term[1], op2);
}

TEST_F(TermTest, MultiplicationDivision) {
  Term term(coeff, {op1, op2});
  std::complex<float> scalar(2.0f, 1.0f);

  Term multiplied = term * scalar;
  EXPECT_EQ(multiplied.coefficient(), coeff * scalar);
  EXPECT_EQ(multiplied.size(), term.size());
}

TEST_F(TermTest, ComparisonOperators) {
  Term term1(coeff, {op1, op2});
  Term term2(coeff, {op1, op2});
  Term term3(coeff, {op2, op1});
  Term term4(std::complex<float>(1.0f, 0.0f), {op1, op2});

  EXPECT_EQ(term1, term2);
  EXPECT_NE(term1, term3);
  EXPECT_NE(term1, term4);
}

// d. Utility function tests
TEST_F(TermTest, ToString) {
  Term term(coeff, {op1, op2});
  std::string expected = "(0.5,-0.5) c+(↑,0)c(↓,1)";
  EXPECT_EQ(term.to_string(), expected);
}

TEST_F(TermTest, HashFunction) {
  Term term1(coeff, {op1, op2});
  Term term2(coeff, {op1, op2});
  Term term3(coeff, {op2, op1});

  EXPECT_EQ(std::hash<Term>{}(term1), std::hash<Term>{}(term2));
  EXPECT_NE(std::hash<Term>{}(term1), std::hash<Term>{}(term3));
}

// e. Edge case tests
TEST_F(TermTest, EmptyOperatorSequence) {
  Term term(coeff);
  EXPECT_EQ(term.size(), 0);
  EXPECT_EQ(term.coefficient(), coeff);
}

TEST_F(TermTest, LargeNumberOfOperators) {
  Term::container_type large_ops(1000, op1);
  Term term(coeff, large_ops);
  EXPECT_EQ(term.size(), 1000);
  EXPECT_EQ(term.coefficient(), coeff);
}

TEST_F(TermTest, IntegrationWithOperator) {
  Term term({0.5f, -0.5f});
  term *= Operator::creation(Operator::Spin::Up, 0);
  term *= Operator::annihilation(Operator::Spin::Down, 1);

  EXPECT_EQ(term.size(), 2);
  EXPECT_EQ(term[0].type(), Operator::Type::Creation);
  EXPECT_EQ(term[0].spin(), Operator::Spin::Up);
  EXPECT_EQ(term[0].orbital(), 0);
  EXPECT_EQ(term[1].type(), Operator::Type::Annihilation);
  EXPECT_EQ(term[1].spin(), Operator::Spin::Down);
  EXPECT_EQ(term[1].orbital(), 1);
  EXPECT_EQ(term.coefficient(), std::complex<float>(0.5f, -0.5f));
}

TEST_F(TermTest, Adjoint) {
  Operator c_up_0 = Operator::creation(Operator::Spin::Up, 0);
  Operator a_down_1 = Operator::annihilation(Operator::Spin::Down, 1);
  std::complex<float> coeff(1.0f, 2.0f);

  Term original(coeff, {c_up_0, a_down_1});
  Term adjoint = original.adjoint();

  EXPECT_EQ(adjoint.coefficient(), std::conj(coeff));

  ASSERT_EQ(adjoint.operators().size(), 2);
  EXPECT_EQ(adjoint[0], a_down_1.adjoint());
  EXPECT_EQ(adjoint[1], c_up_0.adjoint());
}

TEST_F(TermTest, AdjointOfAdjointIsOriginal) {
  Term original(std::complex<float>(1.0f, 2.0f), {op1, op2});
  Term adjoint_of_adjoint = original.adjoint().adjoint();

  EXPECT_EQ(original, adjoint_of_adjoint);
}

}  // namespace
}  // namespace qmutils
