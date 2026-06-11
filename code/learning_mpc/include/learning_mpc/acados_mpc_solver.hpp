/**
 * @file acados_mpc_solver.hpp
 * @brief acados MPC 求解器的 C++ 封装 (静态链接版本)
 *
 * 本类封装了 acados 生成的 C 求解器代码，提供面向对象的 C++ 接口。
 * 使用方式:
 *   1. 先用 Python (generate_diff.py) 生成 C 求解器到 acados_generated/
 *   2. 将生成的 C 文件与本项目一起编译
 *   3. 在 main.cpp 中创建 AcadosMPCSolver 实例并调用 solve()
 *
 * 依赖:
 *   - acados C 库 (libacados)
 *   - acados 生成的 solver 代码 (acados_generated/)
 */

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <stdexcept>

// acados C API 头文件 (需要编译好的 acados 库)
#include "acados_c/ocp_nlp_interface.h"

class AcadosMPCSolver {
public:
    // ================================================================
    // 构造函数 & 析构函数
    // ================================================================

    /**
     * @brief 构造 MPC 求解器
     *
     * @param model_name 模型名称，与 Python diff_model.py 中的一致
     * @param lbx        状态下界 [x_min, y_min, theta_min]
     * @param ubx        状态上界 [x_max, y_max, theta_max]
     * @param lbu        控制量下界 [v_min, omega_min]
     * @param ubu        控制量上界 [v_max, omega_max]
     * @param Q_diag     状态代价对角阵 (3 元素)
     * @param R_diag     控制代价对角阵 (2 元素)
     * @param R_rate_diag 控制变化率代价对角阵 (2 元素)
     */
    AcadosMPCSolver(
        const std::string& model_name,
        const std::vector<double>& lbx,
        const std::vector<double>& ubx,
        const std::vector<double>& lbu,
        const std::vector<double>& ubu,
        const std::vector<double>& Q_diag,
        const std::vector<double>& R_diag,
        const std::vector<double>& R_rate_diag
    );

    ~AcadosMPCSolver();

    // ================================================================
    // 核心求解接口
    // ================================================================

    /** @brief 执行一次 MPC 求解 (SQP_RTI 仅 1 次迭代) */
    int solve();

    /** @brief 设置当前初始状态 x0 = [x, y, theta] */
    void setInitialState(const std::vector<double>& x0);

    /** @brief 设置第 stage 步的参考状态 (用于代价函数) */
    void setReference(int stage, const std::vector<double>& yref);

    /** @brief 设置第 stage 步的外部参数 (例如上一时刻的控制量) */
    void setParameter(int stage, const std::vector<double>& p);

    // ================================================================
    // 结果获取接口
    // ================================================================

    /** @brief 获取第 stage 步的状态 [x, y, theta] */
    std::vector<double> getState(int stage);

    /** @brief 获取第 stage 步的控制量 [v, omega] */
    std::vector<double> getControl(int stage);

    /** @brief 获取最近一次求解的耗时 [ms] */
    double getSolveTime() const { return solve_time_ms; }

    /** @brief 获取最近一次求解的 SQP 迭代次数 */
    int getSQPIterations() const { return sqp_iter; }

    /** @brief 打印求解器维度信息 */
    void printSolverInfo() const;

    // ================================================================
    // 维度查询
    // ================================================================

    int getNX() const { return nx; }
    int getNU() const { return nu; }
    int getN()  const { return N; }

private:
    // ---- acados 内部对象 ----
    ocp_nlp_config*  nlp_config;    // NLP 求解器配置
    ocp_nlp_dims*    nlp_dims;      // NLP 维度
    ocp_nlp_in*      nlp_in;        // NLP 输入 (初始状态、参考、参数)
    ocp_nlp_out*     nlp_out;       // NLP 输出 (状态轨迹、控制序列)
    ocp_nlp_solver*  nlp_solver;    // NLP 求解器
    void*            nlp_opts;      // NLP 求解器选项 (opaque)

    // ---- 问题维度 (从生成代码中读取) ----
    int nx;   // 状态维度 = 3
    int nu;   // 控制维度 = 2
    int N;    // 预测步数 = 30

    // ---- 求解统计 ----
    double solve_time_ms;   // 最近一次求解耗时 [ms]
    int    sqp_iter;        // 最近一次 SQP 迭代次数

    // ---- 参数存储 (每步的外部参数) ----
    std::vector<std::vector<double>> params_storage;
};
