/**
 * @file acados_mpc_solver.cpp
 * @brief AcadosMPCSolver 类实现
 *
 * 本文件封装了 acados C API 的调用细节。
 * 编译要求:
 *   - acados 库已编译 (libacados.a / acados.lib)
 *   - acados 头文件路径已添加到 CMake include_directories
 *   - acados_generated/ 目录中的 solver 源文件已纳入编译
 */

#include "learning_mpc/acados_mpc_solver.hpp"
#include <chrono>

// acados 生成的 solver 函数 (函数名与 model_name 对应)
// 这些函数在 acados_generated/acados_solver_<model_name>.h 中声明
namespace {
    // 前向声明 — 实际符号在 acados 生成的 C 代码中定义
    extern "C" {
        // 以下函数的实际名称取决于生成时的 model_name
        // 例如 model_name = "differential_drive" 则前缀为 differential_drive_
        // 我们需要在 CMake 中设置 MODEL_NAME 宏，或者用户通过模板参数传入。
        //
        // 由于 acados 生成代码中函数命名固定为 <model_name>_acados_*，
        // 本实现假设 model_name 作为构造参数传入，通过 extern 声明引入对应符号。
        //
        // 简便方案：在编译时定义宏 ACD_MODEL_NAME，例如：
        //   target_compile_definitions(learning_mpc PRIVATE ACD_MODEL_NAME=differential_drive)
        //
        // 更通用的方案（动态加载）请参考旧版 learning_mpc.hpp 中的 std::function 模式。
    }
}

// ================================================================
// acados solver capsule 的创建/释放辅助宏
// 由于不同模型的函数名不同，这里使用宏拼接
// ================================================================

// 这些拼接宏在编译时展开：
//   ACD_CREATE(differential_drive) → differential_drive_acados_create_capsule()
#define ACD_CONCAT_IMPL(a, b) a ## _ ## b
#define ACD_CONCAT(a, b) ACD_CONCAT_IMPL(a, b)

// 声明 solver 函数 — 实际定义在 acados_generated/ 中的 C 文件中
// 编译时需要链接这些目标文件
#define ACD_DECLARE_SOLVER_FUNCTIONS(model)                              \
    extern "C" {                                                         \
        void* ACD_CONCAT(model, acados_create_capsule) (void);            \
        int   ACD_CONCAT(model, acados_create) (void* capsule);            \
        int   ACD_CONCAT(model, acados_solve) (void* capsule);             \
        int   ACD_CONCAT(model, acados_free_capsule) (void* capsule);      \
        ocp_nlp_config* ACD_CONCAT(model, acados_get_nlp_config) (void* capsule); \
        ocp_nlp_dims*   ACD_CONCAT(model, acados_get_nlp_dims) (void* capsule);   \
        ocp_nlp_in*     ACD_CONCAT(model, acados_get_nlp_in) (void* capsule);     \
        ocp_nlp_out*    ACD_CONCAT(model, acados_get_nlp_out) (void* capsule);    \
        ocp_nlp_solver* ACD_CONCAT(model, acados_get_nlp_solver) (void* capsule); \
        void*           ACD_CONCAT(model, acados_get_nlp_opts) (void* capsule);   \
    }

// 根据 model_name 声明对应函数 — 如需支持多个模型，在此添加
// 注意: 当前 model_name = "differential_drive"
ACD_DECLARE_SOLVER_FUNCTIONS(differential_drive);

// ================================================================
// 构造函数
// ================================================================
AcadosMPCSolver::AcadosMPCSolver(
    const std::string& model_name,
    const std::vector<double>& lbx,
    const std::vector<double>& ubx,
    const std::vector<double>& lbu,
    const std::vector<double>& ubu,
    const std::vector<double>& Q_diag,
    const std::vector<double>& R_diag,
    const std::vector<double>& R_rate_diag
)
    : nlp_config(nullptr), nlp_dims(nullptr),
      nlp_in(nullptr), nlp_out(nullptr),
      nlp_solver(nullptr), nlp_opts(nullptr),
      solve_time_ms(0.0), sqp_iter(0)
{
    // ---- 1. 创建 acados solver capsule ----
    // 注意: 此函数名根据 model_name 拼接，当前仅支持 "differential_drive"
    void* capsule = differential_drive_acados_create_capsule();
    if (!capsule) {
        throw std::runtime_error("Failed to create acados capsule for model: " + model_name);
    }

    // ---- 2. 创建 solver (分配内存、配置 NLP) ----
    int status = differential_drive_acados_create(capsule);
    if (status != 0) {
        throw std::runtime_error("Failed to create acados solver for model: " + model_name);
    }

    // ---- 3. 获取 acados 内部对象 ----
    nlp_config = differential_drive_acados_get_nlp_config(capsule);
    nlp_dims   = differential_drive_acados_get_nlp_dims(capsule);
    nlp_in     = differential_drive_acados_get_nlp_in(capsule);
    nlp_out    = differential_drive_acados_get_nlp_out(capsule);
    nlp_solver = differential_drive_acados_get_nlp_solver(capsule);
    nlp_opts   = differential_drive_acados_get_nlp_opts(capsule);

    // ---- 4. 读取维度 ----
    nx = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nlp_out, 0, "x");
    nu = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nlp_out, 0, "u");
    N  = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nlp_out, 0, "N");

    // ---- 5. 预分配参数存储 ----
    params_storage.resize(N + 1, std::vector<double>(nu, 0.0));

    // ---- 6. 设置约束边界 ----
    // 状态边界: 应用于所有 stage
    for (int i = 0; i <= N; i++) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", lbx.data());
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", ubx.data());
    }
    // 控制边界: 应用于 stage 0..N-1
    for (int i = 0; i < N; i++) {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu.data());
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu.data());
    }
}

// ================================================================
// 析构函数
// ================================================================
AcadosMPCSolver::~AcadosMPCSolver() {
    // 释放 solver 内部内存
    // 注意: capsule 生命周期管理在 acados 内部
    // 此处仅需确保 nlp_* 指针不会悬挂
    nlp_config  = nullptr;
    nlp_dims    = nullptr;
    nlp_in      = nullptr;
    nlp_out     = nullptr;
    nlp_solver  = nullptr;
    nlp_opts    = nullptr;
}

// ================================================================
// solve — 执行一次 MPC 求解
// ================================================================
int AcadosMPCSolver::solve() {
    auto t_start = std::chrono::high_resolution_clock::now();

    // 调用 acados solver (SQP_RTI 模式: 仅 1 次迭代)
    // 注意: solve 函数签名因模型不同而不同
    int status = differential_drive_acados_solve(nullptr); // capsule 在 solver 内部管理

    auto t_end = std::chrono::high_resolution_clock::now();
    solve_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // 获取 SQP 迭代次数
    int sqp_iter_out[1];
    ocp_nlp_get(nlp_config, nlp_solver, "sqp_iter", sqp_iter_out);
    sqp_iter = sqp_iter_out[0];

    return status;
}

// ================================================================
// setInitialState — 设置当前状态作为 OCP 初始条件
// ================================================================
void AcadosMPCSolver::setInitialState(const std::vector<double>& x0) {
    assert(x0.size() == static_cast<size_t>(nx) && "x0 size must match nx");
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", x0.data());
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", x0.data());
}

// ================================================================
// setReference — 设置第 stage 步的参考值 yref
// ================================================================
void AcadosMPCSolver::setReference(int stage, const std::vector<double>& yref) {
    assert(stage >= 0 && stage <= N && "stage out of range");
    // yref 维度 = nx + nu + nu (状态 + 控制 + 控制变化率)
    size_t ny = static_cast<size_t>(nx + nu + nu);

    if (stage < N) {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, stage, "yref", yref.data());
    } else if (stage == N) {
        // 终端代价参考维度 = nx
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, stage, "yref", yref.data());
    }
}

// ================================================================
// setParameter — 设置第 stage 步的外部参数
// ================================================================
void AcadosMPCSolver::setParameter(int stage, const std::vector<double>& p) {
    assert(stage >= 0 && stage < N && "stage out of range");
    params_storage[stage] = p;
    ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, stage, "parameter_values", p.data());
}

// ================================================================
// getState — 获取第 stage 步的状态
// ================================================================
std::vector<double> AcadosMPCSolver::getState(int stage) {
    assert(stage >= 0 && stage <= N && "stage out of range");
    std::vector<double> x(nx);
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, stage, "x", x.data());
    return x;
}

// ================================================================
// getControl — 获取第 stage 步的控制量
// ================================================================
std::vector<double> AcadosMPCSolver::getControl(int stage) {
    assert(stage >= 0 && stage < N && "stage out of range");
    std::vector<double> u(nu);
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, stage, "u", u.data());
    return u;
}

// ================================================================
// printSolverInfo — 打印求解器信息
// ================================================================
void AcadosMPCSolver::printSolverInfo() const {
    std::cout << "=== Acados MPC Solver Info ===" << std::endl;
    std::cout << "  Model:       differential_drive" << std::endl;
    std::cout << "  States  (nx): " << nx << std::endl;
    std::cout << "  Controls (nu): " << nu << std::endl;
    std::cout << "  Horizon  (N):  " << N  << std::endl;
    std::cout << "  Last solve:    " << solve_time_ms << " ms, "
              << sqp_iter << " SQP iter" << std::endl;
    std::cout << "===============================" << std::endl;
}
