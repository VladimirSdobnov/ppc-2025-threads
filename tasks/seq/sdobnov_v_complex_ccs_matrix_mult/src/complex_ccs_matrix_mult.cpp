#include "seq/sdobnov_v_complex_ccs_matrix_mult/include/complex_ccs_matrix_mult.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

void sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS::AddValue(int col, int row, const std::complex<double>& value) {
  if (col < 0 || col >= cols || row < 0 || row >= rows) {
    throw std::out_of_range("Invalid row or column index in AddValue()");
  }
  if (std::abs(value) < 1e-10) {
    return;
  }

  int insert_pos = col_p[col + 1];

  values.insert(values.begin() + insert_pos, value);
  row_i.insert(row_i.begin() + insert_pos, row);

  for (int i = col + 1; i <= cols; ++i) {
    col_p[i]++;
  }
}

bool sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS::operator==(const SparseMatrixCCS& other) const {
  if (rows != other.rows || cols != other.cols) {
    return false;
  }
  if (col_p != other.col_p || row_i != other.row_i) {
    return false;
  }
  if (values.size() != other.values.size()) {
    return false;
  }

  for (size_t i = 0; i < values.size(); ++i) {
    if (std::abs(values[i] - other.values[i]) > 1e-6) {
      return false;
    }
  }
  return true;
}

sdobnov_v_complex_ccs_matrix_mult::SparseMatrixCCS sdobnov_v_complex_ccs_matrix_mult::GenerateRandomMatrix(
    int rows, int cols, double density, int seed = 42) {
  if (density < 0.0 || density > 1.0) {
    throw std::invalid_argument("Density must be between 0.0 and 1.0");
  }

  SparseMatrixCCS mat(rows, cols);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> prob(0.0, 1.0);
  std::uniform_real_distribution<double> val_dist(-10.0, 10.0);

  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      if (prob(gen) < density) {
        std::complex<double> value(val_dist(gen), val_dist(gen));
        mat.AddValue(col, row, value);
      }
    }
  }

  return mat;
}

bool sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult::PreProcessingImpl() {
  M1_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[0]);
  M2_ = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[1]);
  Res_ = reinterpret_cast<SparseMatrixCCS*>(task_data->outputs[0]);
  return true;
}

bool sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult::ValidationImpl() {
  int m1_cols_n = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[0])->cols;
  int m2_rows_n = reinterpret_cast<SparseMatrixCCS*>(task_data->inputs[1])->rows;
  return (m1_cols_n == m2_rows_n);
}

bool sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult::RunImpl() {
  int rows_m1 = M1_->rows;
  int cols_m2 = M2_->cols;

  Res_->rows = rows_m1;
  Res_->cols = cols_m2;
  Res_->values.clear();
  Res_->row_i.clear();
  Res_->col_p.assign(cols_m2 + 1, 0);

  std::vector<std::complex<double>> row_buffer(rows_m1, {0.0, 0.0});

  for (int col_m2 = 0; col_m2 < cols_m2; ++col_m2) {
    std::vector<int> rows_in_col;
    std::vector<std::complex<double>> vals_in_col;

    for (int idx_m2 = M2_->col_p[col_m2]; idx_m2 < M2_->col_p[col_m2 + 1]; ++idx_m2) {
      int row_m2 = M2_->row_i[idx_m2];
      std::complex<double> val_m2 = M2_->values[idx_m2];

      for (int idx_m1 = M1_->col_p[row_m2]; idx_m1 < M1_->col_p[row_m2 + 1]; ++idx_m1) {
        int row_m1 = M1_->row_i[idx_m1];
        std::complex<double> val_m1 = M1_->values[idx_m1];

        row_buffer[row_m1] += val_m1 * val_m2;
      }
    }

    for (int row = 0; row < rows_m1; ++row) {
      if (std::abs(row_buffer[row]) > 1e-10) {
        rows_in_col.push_back(row);
        vals_in_col.push_back(row_buffer[row]);
        row_buffer[row] = {0.0, 0.0};
      }
    }

    Res_->col_p[col_m2 + 1] = Res_->col_p[col_m2] + static_cast<int>(vals_in_col.size());
    Res_->values.insert(Res_->values.end(), vals_in_col.begin(), vals_in_col.end());
    Res_->row_i.insert(Res_->row_i.end(), rows_in_col.begin(), rows_in_col.end());
  }

  return true;
}

bool sdobnov_v_complex_ccs_matrix_mult::SeqComplexCcsMatrixMult::PostProcessingImpl() { return true; }
