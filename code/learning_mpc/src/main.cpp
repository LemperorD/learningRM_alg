/**
 * @file main.cpp
 * @brief 差速驱动底盘 MPC 闭环控制 Demo (C++ / acados)
 *
 * 功能:
 *   使用 acados 生成的 NMPC 求解器，对差速驱动底盘进行实时闭环控制。
 *   机器人从 (0, 0, 0) 出发，目标点为 (1.0, 0.5, 0.0)，
 *   每步求解 MPC → 施加第一步控制 → 仿真一步 → 重复。
 *
 * 编译前提:
 *   1. 已编译 acados 库
 *   2. 已运行 python/generate_diff.py 生成 acados_generated/
 *   3. CMake 中配置了 acados 的 include 和 lib 路径
 *
 * 用法:
 *   ./learning_mpc
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

#include "learning_mpc/acados_mpc_solver.hpp"

// ================================================================
// 问题参数 (与 generate_diff.py 保持一致)
// ================================================================
constexpr double Tf      = 1.0;       // 预测时域 [s]
constexpr int    N_STEPS = 30;        // 离散步数
constexpr double DT      = Tf / N_STEPS;  // 控制周期 [s]

const std::vector<double> LBX  = {-10.0, -10.0, -3.14159};
const std::vector<double> UBX  = { 10.0,  10.0,  3.14159};
const std::vector<double> LBU  = {-1.5, -1.5};
const std::vector<double> UBU  = { 1.5,  1.5};
const std::vector<double> Q_DIAG = {25.0, 25.0, 25.0};
const std::vector<double> R_DIAG = { 0.1,  0.1};
const std::vector<double> R_RATE_DIAG = {2.0, 2.0};

// 目标状态
const std::vector<double> TARGET = {1.0, 0.5, 0.0};

// 仿真参数
constexpr int    MAX_SIM_STEPS  = 200;
constexpr double GOAL_TOLERANCE = 0.01;

// ================================================================
// RK4 积分一步 (unicycle dynamics)
// ================================================================
std::vector<double> rk4_step(const std::vector<double>& x,
                              const std::vector<double>& u, double dt) {
    auto f = [](const std::vector<double>& x_, const std::vector<double>& u_) {
        return std::vector<double>{
            u_[0] * std::cos(x_[2]),    // dx/dt = v * cos(theta)
            u_[0] * std::sin(x_[2]),    // dy/dt = v * sin(theta)
            u_[1]                        // dtheta/dt = omega
        };
    };

    auto add  = [](const std::vector<double>& a, const std::vector<double>& b) {
        return std::vector<double>{a[0]+b[0], a[1]+b[1], a[2]+b[2]};
    };
    auto scale = [](double s, const std::vector<double>& v) {
        return std::vector<double>{s*v[0], s*v[1], s*v[2]};
    };

    auto k1 = f(x, u);
    auto k2 = f(add(x, scale(dt/2, k1)), u);
    auto k3 = f(add(x, scale(dt/2, k2)), u);
    auto k4 = f(add(x, scale(dt, k3)), u);

    return add(x, scale(dt/6, add(add(k1, scale(2, k2)), add(scale(2, k3), k4))));
}

// ================================================================
// 计算位置误差
// ================================================================
double position_error(const std::vector<double>& x, const std::vector<double>& target) {
    double dx = x[0] - target[0];
    double dy = x[1] - target[1];
    return std::sqrt(dx*dx + dy*dy);
}

// ================================================================
// main — MPC 闭环控制 Demo
// ================================================================
int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Differential Drive MPC Demo (acados)" << std::endl;
    std::cout << "================================================" << std::endl;

    // ---- 1. 创建 MPC 求解器 ----
    std::cout << "\n[1] Creating MPC solver..." << std::endl;

    AcadosMPCSolver solver("differential_drive", LBX, UBX, LBU, UBU,
                           Q_DIAG, R_DIAG, R_RATE_DIAG);
    solver.printSolverInfo();

    // ---- 2. 设置参考轨迹 (全部指向目标) ----
    std::cout << "\n[2] Setting reference trajectory..." << std::endl;
    // yref 维度 = nx + nu + nu = 3 + 2 + 2 = 7
    // yref = [x_ref, y_ref, theta_ref, v_ref, omega_ref, v_rate_ref, omega_rate_ref]
    std::vector<double> yref(7, 0.0);
    yref[0] = TARGET[0];   // x_ref
    yref[1] = TARGET[1];   // y_ref
    yref[2] = TARGET[2];   // theta_ref
    // v_ref, omega_ref, v_rate_ref, omega_rate_ref 保持 0

    for (int i = 0; i <= solver.getN(); i++) {
        if (i < solver.getN()) {
            solver.setReference(i, yref);
        } else {
            // 终端参考: 仅状态 [x_ref, y_ref, theta_ref]
            std::vector<double> yref_e = {TARGET[0], TARGET[1], TARGET[2]};
            solver.setReference(i, yref_e);
        }
    }

    // ---- 3. 闭环仿真 ----
    std::cout << "\n[3] Starting closed-loop simulation..." << std::endl;
    std::cout << "    Target: x=" << TARGET[0] << ", y=" << TARGET[1]
              << ", theta=" << TARGET[2] << std::endl;

    std::vector<double> x_current = {0.0, 0.0, 0.0};  // 初始状态
    std::vector<double> u_prev    = {0.0, 0.0};        // 上一控制量

    for (int step = 0; step < MAX_SIM_STEPS; step++) {
        // 设置初始状态
        solver.setInitialState(x_current);

        // 设置参数 (上一控制量，用于控制变化率惩罚)
        solver.setParameter(0, u_prev);

        // 求解 MPC
        int status = solver.solve();
        if (status != 0) {
            std::cerr << "  [ERROR] Solver failed at step " << step << std::endl;
            break;
        }

        // 获取最优控制量 (第一步)
        std::vector<double> u_opt = solver.getControl(0);

        // 施加控制 + 仿真一步
        std::vector<double> x_next = rk4_step(x_current, u_opt, DT);

        // 打印信息
        if (step % 20 == 0 || step < 5) {
            printf("  Step %3d | u=[%+.3f, %+.3f] | x=[%+.3f, %+.3f, %+.3f] | "
                   "t=%.1fms\n",
                   step, u_opt[0], u_opt[1],
                   x_current[0], x_current[1], x_current[2],
                   solver.getSolveTime());
        }

        // 更新
        x_current = x_next;
        u_prev    = u_opt;

        // 检查到达目标
        double pos_err = position_error(x_current, TARGET);
        double theta_err = std::abs(x_current[2] - TARGET[2]);
        if (pos_err < GOAL_TOLERANCE && theta_err < 0.05) {
            printf("\n  [OK] Goal reached at step %d! pos_err=%.4fm, theta_err=%.4frad\n",
                   step + 1, pos_err, theta_err);
            break;
        }
    }

    std::cout << "\n[4] Simulation finished." << std::endl;
    return 0;
}
