#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "seq/sdobnov_v_complex_ccs_matrix_mult/include/complex_ccs_matrix_mult.hpp"

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, test_pipeline_run) {
  int rows = 200000;
  int cols = 200000;
  auto a = sdobnov_v_complex_ccs_matrix_mult_seq::GenerateRandomMatrix(rows, cols, 0.0001, 123);
  auto b = sdobnov_v_complex_ccs_matrix_mult_seq::GenerateRandomMatrix(cols, 1, 0.5, 321);
  sdobnov_v_complex_ccs_matrix_mult_seq::SparseMatrixCCS c(rows, 1);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&a), reinterpret_cast<uint8_t*>(&b)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&c)};

  auto test_task = std::make_shared<sdobnov_v_complex_ccs_matrix_mult_seq::SeqComplexCcsMatrixMult>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(test_task->ValidationImpl());
}

TEST(sdobnov_v_complex_ccs_matrix_mult_seq, test_task_run) {
  int rows = 200000;
  int cols = 200000;
  auto a = sdobnov_v_complex_ccs_matrix_mult_seq::GenerateRandomMatrix(rows, cols, 0.0001, 456);
  auto b = sdobnov_v_complex_ccs_matrix_mult_seq::GenerateRandomMatrix(cols, 1, 0.5, 654);
  sdobnov_v_complex_ccs_matrix_mult_seq::SparseMatrixCCS c(rows, 1);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs = {reinterpret_cast<uint8_t*>(&a), reinterpret_cast<uint8_t*>(&b)};
  task_data->outputs = {reinterpret_cast<uint8_t*>(&c)};

  auto test_task = std::make_shared<sdobnov_v_complex_ccs_matrix_mult_seq::SeqComplexCcsMatrixMult>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_TRUE(test_task->ValidationImpl());
}