#pragma once

#include <complex>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sdobnov_v_complex_ccs_matrix_mult {
struct SparseMatrixCCS {
  int rows;
  int cols;
  std::vector<std::complex<double>> values;
  std::vector<int> row_i;
  std::vector<int> col_p;

  SparseMatrixCCS(int r = 0, int c = 0) : rows(r), cols(c), col_p(c + 1, 0) {}

  void AddValue(int col, int row, const std::complex<double>& value);

  bool operator==(const SparseMatrixCCS& other) const;
};

class SeqComplexCcsMatrixMult : public ppc::core::Task {
 public:
  explicit SeqComplexCcsMatrixMult(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  SparseMatrixCCS *M1_, *M2_, *Res_;
};

SparseMatrixCCS GenerateRandomMatrix(int rows, int cols, double density, int seed);
}  // namespace sdobnov_v_complex_ccs_matrix_mult
