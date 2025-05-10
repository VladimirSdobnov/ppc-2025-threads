#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/sdobnov_v_complex_ccs_matrix_mult/include/complex_ccs_matrix_mult.hpp"

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, Multiply_IncompatibleDimensions) {
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m1(2, 3);
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m2(4, 5);
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS result(2, 5);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&m1), reinterpret_cast<uint8_t*>(&m2)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&result)};

  sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult task(task_data);
  ASSERT_FALSE(task.ValidationImpl());
}

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, Multiply_EmptyMatrices) {
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m1(2, 3);
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m2(3, 4);
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS result(2, 4);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&m1), reinterpret_cast<uint8_t*>(&m2)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&result)};

  sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  ASSERT_EQ(result.values.size(), static_cast<size_t>(0));
  ASSERT_EQ(result.row_i.size(), static_cast<size_t>(0));
  for (int p : result.col_p) {
    ASSERT_EQ(p, 0);
  }
}

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, Multiply_1x1) {
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m1(1, 1);
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m2(1, 1);
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS result(1, 1);
  m1.AddValue(0, 0, {0.0, 1.0});
  m2.AddValue(0, 0, {0.0, -1.0});

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&m1), reinterpret_cast<uint8_t*>(&m2)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&result)};

  sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult task(task_data);

  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS expected(1, 1);
  expected.AddValue(0, 0, {1.0, 0.0});

  ASSERT_EQ(result, expected);
}

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, Multiply_2x3_3x2) {
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS a(2, 3);
  a.AddValue(0, 0, {1.0, 0.0});
  a.AddValue(1, 1, {0.0, 1.0});
  a.AddValue(2, 1, {1.0, -1.0});
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS b(3, 2);
  b.AddValue(0, 0, {0.0, 1.0});
  b.AddValue(1, 0, {2.0, 0.0});
  b.AddValue(0, 1, {1.0, 0.0});
  b.AddValue(0, 2, {1.0, 1.0});
  b.AddValue(1, 2, {0.0, 2.0});
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS result(2, 2);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&a), reinterpret_cast<uint8_t*>(&b)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&result)};

  sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS expected(2, 2);
  expected.AddValue(0, 0, {0.0, 1.0});
  expected.AddValue(1, 0, {2.0, 0.0});
  expected.AddValue(0, 1, {2.0, 1.0});
  expected.AddValue(1, 1, {2.0, 2.0});

  ASSERT_EQ(result, expected);
}

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, Multiply_RandomSparseMatrices) {
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m1 =
      sdobnov_v_complex_ccs_matrix_mult::GenerateRandomMatrix(10, 5, 0.1, 1);
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS m2 =
      sdobnov_v_complex_ccs_matrix_mult::GenerateRandomMatrix(5, 8, 0.1, 2);
  sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS result(10, 8);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&m1), reinterpret_cast<uint8_t*>(&m2)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&result)};

  sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  task.PreProcessingImpl();
  task.RunImpl();
  task.PostProcessingImpl();

  ASSERT_EQ(result.col_p.size(), static_cast<size_t>(9));
  ASSERT_EQ(result.values.size(), result.row_i.size());
}