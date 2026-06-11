#!/usr/bin/env python3
"""
差速驱动底盘 NMPC 闭环仿真与测试
==================================

功能:
    使用 CasADi 内置 NLP 求解器 (IPOPT) 对差速驱动底盘进行非线性 MPC
    闭环仿真。该脚本独立于 acados，用于在生成 C 代码之前验证 MPC
    的数学公式、权重参数和约束配置是否正确。

仿真场景:
    机器人从原点 (0, 0, -0.5 rad) 出发，目标点位于 (0.5, 0.5, 0 rad)。
    MPC 控制器每步求解一个有限时域最优控制问题，施加第一步控制后更新
    状态，重复直至到达目标或达到最大步数。

输出:
    - 控制台打印每一步的求解信息
    - 弹出 / 保存三张图:
      1. XY 平面轨迹 (鸟瞰图)
      2. 状态随时间变化曲线 (x, y, theta)
      3. 控制量随时间变化曲线 (v, omega)

依赖:
    - casadi, numpy, matplotlib
"""

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


# ============================================================
# 全局配置 — 与 generate_diff.py 保持完全一致
# ============================================================
Tf = 1.0              # 预测时域 [s]
N  = 30               # 离散化步数
dt = Tf / N           # 单步时长 ≈ 0.033 s

# 代价权重
Q_DIAG      = np.array([25.0, 25.0, 25.0])   # 状态跟踪 [x, y, theta]
R_DIAG      = np.array([0.1, 0.1])            # 控制抑制 [v, omega]
R_RATE_DIAG = np.array([2.0, 2.0])            # 控制变化率抑制 [Δv, Δomega]

# 状态边界
X_MIN = np.array([-10.0, -10.0, -np.pi])
X_MAX = np.array([10.0,  10.0,   np.pi])

# 控制量边界
U_MIN = np.array([-1.5, -1.5])
U_MAX = np.array([1.5,  1.5])

# 仿真参数
MAX_SIM_STEPS    = 200          # 最大仿真步数
GOAL_TOLERANCE   = 0.01        # 到达目标的位置容差 [m]
GOAL_THETA_TOL   = 0.05        # 到达目标的角度容差 [rad]
TARGET           = np.array([1.0, 0.5, 0.0])  # 目标状态 [x, y, theta]


def build_mpc_solver() -> ca.Function:
    """
    用 CasADi Opti 构建与 acados 等价的 NMPC 问题，返回一个求解函数。

    OCP 公式 (与 generate_diff.py 完全一致):
        min  Σ ||x_k - x_ref||²_Q + ||u_k||²_R + ||u_k - u_{k-1}||²_{R_rate}
             + ||x_N - x_ref||²_Q
        s.t. x_0     = x_current             (初始状态)
             x_{k+1} = f(x_k, u_k)           (动力学，RK4 离散)
             x_min ≤ x_k ≤ x_max              (状态约束)
             u_min ≤ u_k ≤ u_max              (控制约束)

    Returns
    -------
    solve : casadi.Function
        输入: x0 (3,), u_prev (2,), x_ref (3,)
        输出: u_opt (2*N,), x_pred (3*(N+1),), cost
    """
    opti = ca.Opti()

    # ---- 决策变量 ----
    X = opti.variable(3, N + 1)    # 状态轨迹 [x, y, theta] × (N+1)
    U = opti.variable(2, N)        # 控制序列 [v, omega] × N

    # ---- 参数 ----
    x0_param  = opti.parameter(3)          # 当前状态
    u_prev_p  = opti.parameter(2)          # 上一步控制 (用于变化率惩罚)
    x_ref_p   = opti.parameter(3)          # 参考/目标状态

    # ---- 目标函数 ----
    cost = 0
    Q = np.diag(Q_DIAG)
    R = np.diag(R_DIAG)
    R_rate = np.diag(R_RATE_DIAG)

    for k in range(N):
        x_err  = X[:, k] - x_ref_p
        cost  += x_err.T @ Q @ x_err                    # 状态跟踪代价
        cost  += U[:, k].T @ R @ U[:, k]                # 控制量代价
        if k == 0:
            cost += (U[:, 0] - u_prev_p).T @ R_rate @ (U[:, 0] - u_prev_p)
        else:
            cost += (U[:, k] - U[:, k-1]).T @ R_rate @ (U[:, k] - U[:, k-1])

    # 终端代价
    x_err_N = X[:, N] - x_ref_p
    cost   += x_err_N.T @ Q @ x_err_N

    opti.minimize(cost)

    # ---- 动力学约束 (RK4 离散化) ----
    def dynamics_rk4(xk, uk, h):
        """RK4 积分一步: x_{k+1} = x_k + h * f_RK4(x_k, u_k)"""
        f = lambda x_, u_: ca.vertcat(
            u_[0] * ca.cos(x_[2]),    # dx/dt = v * cos(theta)
            u_[0] * ca.sin(x_[2]),    # dy/dt = v * sin(theta)
            u_[1]                      # dtheta/dt = omega
        )
        k1 = f(xk,       uk)
        k2 = f(xk + h/2 * k1, uk)
        k3 = f(xk + h/2 * k2, uk)
        k4 = f(xk + h   * k3, uk)
        return xk + h / 6 * (k1 + 2*k2 + 2*k3 + k4)

    for k in range(N):
        x_next = dynamics_rk4(X[:, k], U[:, k], dt)
        opti.subject_to(X[:, k+1] == x_next)

    # ---- 状态边界 ----
    for k in range(N + 1):
        opti.subject_to(opti.bounded(X_MIN, X[:, k], X_MAX))

    # ---- 控制量边界 ----
    for k in range(N):
        opti.subject_to(opti.bounded(U_MIN, U[:, k], U_MAX))

    # ---- 初始状态约束 ----
    opti.subject_to(X[:, 0] == x0_param)

    # ---- 终端等式约束 (强制收敛到目标，避免 unicycle 局部极小值) ----
    opti.subject_to(X[:, N] == x_ref_p)

    # ---- 求解器配置 ----
    opts = {
        'ipopt.print_level': 0,           # 关闭 IPOPT 详细输出
        'ipopt.sb': 'yes',                # 抑制 banner
        'ipopt.max_iter': 200,
        'print_time': False,
    }
    opti.solver('ipopt', opts)

    # ---- 构建返回函数 ----
    solve = opti.to_function('mpc_solve', [x0_param, u_prev_p, x_ref_p],
                             [U[:, 0], X, cost, U])
    return solve


def run_closed_loop_simulation(solve_func: ca.Function):
    """
    运行 MPC 闭环仿真。

    Parameters
    ----------
    solve_func : casadi.Function
        由 build_mpc_solver() 返回的 MPC 求解函数。
    """
    # 初始状态
    x_current = np.array([0.0, 0.0, 0.0])   # 初始位置原点，朝向+x轴
    u_prev    = np.array([0.0, 0.0])          # 初始时刻机器人静止

    # 记录轨迹
    x_history = [x_current.copy()]
    u_history = [u_prev.copy()]
    cost_history = []
    solve_time_history = []

    print("=" * 65)
    print("  NMPC Closed-loop Simulation")
    print(f"  Init: x={x_current[0]:.2f}, y={x_current[1]:.2f}, theta={x_current[2]:.2f}")
    print(f"  Goal: x={TARGET[0]:.2f}, y={TARGET[1]:.2f}, theta={TARGET[2]:.2f}")
    print("=" * 65)

    for step in range(MAX_SIM_STEPS):
        # 求解 MPC
        import time
        t0 = time.perf_counter()
        result = solve_func(x_current, u_prev, TARGET)
        t_solve = time.perf_counter() - t0

        u_opt   = np.array(result[0]).flatten()     # 最优第一步控制
        x_pred  = np.array(result[1])                # 预测轨迹
        cost_val = float(result[2])
        U_pred  = np.array(result[3])                # 全部控制序列

        solve_time_history.append(t_solve)
        cost_history.append(cost_val)

        # 施加控制 + 仿真一步 (RK4)
        def rk4_step(xk, uk, h):
            f = lambda x_, u_: np.array([
                u_[0] * np.cos(x_[2]),
                u_[0] * np.sin(x_[2]),
                u_[1]
            ])
            k1 = f(xk,       uk)
            k2 = f(xk + h/2 * k1, uk)
            k3 = f(xk + h/2 * k2, uk)
            k4 = f(xk + h   * k3, uk)
            return xk + h / 6 * (k1 + 2*k2 + 2*k3 + k4)

        x_next = rk4_step(x_current, u_opt, dt)

        # 记录
        x_history.append(x_next.copy())
        u_history.append(u_opt.copy())

        # 打印信息
        if step % 20 == 0 or step < 5:
            print(f"  Step {step:3d} | u=[{u_opt[0]:+.3f}, {u_opt[1]:+.3f}] | "
                  f"x=[{x_current[0]:+.3f}, {x_current[1]:+.3f}, "
                  f"{x_current[2]:+.3f}] | "
                  f"cost={cost_val:.4f} | t={t_solve*1000:.1f}ms")

        # 更新
        x_current = x_next
        u_prev    = u_opt

        # Check if goal reached
        pos_err = np.linalg.norm(x_current[:2] - TARGET[:2])
        theta_err = abs(x_current[2] - TARGET[2])
        if pos_err < GOAL_TOLERANCE and theta_err < GOAL_THETA_TOL:
            print(f"\n  [OK] Goal reached at step {step+1}! pos_err={pos_err:.4f}m, "
                  f"theta_err={theta_err:.4f}rad")
            break
    else:
        pos_err = np.linalg.norm(x_current[:2] - TARGET[:2])
        print(f"\n  [WARN] Max steps {MAX_SIM_STEPS} reached, pos_err={pos_err:.4f}m")

    avg_time = np.mean(solve_time_history) * 1000
    print(f"\n  Avg solve time: {avg_time:.1f} ms/step")
    print(f"  Total sim steps: {len(x_history)-1}")

    return (np.array(x_history), np.array(u_history),
            np.array(cost_history), np.array(solve_time_history))


def plot_results(x_hist, u_hist, cost_hist, time_hist):
    """
    生成仿真结果图:
      1. XY 平面轨迹 (鸟瞰图)
      2. 状态随时间变化 (x, y, theta)
      3. 控制量随时间变化 (v, omega)

    Parameters
    ----------
    x_hist   : np.ndarray, shape (M, 3)   — 状态历史
    u_hist   : np.ndarray, shape (M, 2)   — 控制量历史
    cost_hist: np.ndarray, shape (M-1,)   — 每步代价
    time_hist: np.ndarray, shape (M-1,)   — 每步求解时间 [s]
    """
    t_sim = np.arange(len(x_hist)) * dt

    # ---- 图1: XY 平面轨迹 ----
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    ax1.plot(x_hist[:, 0], x_hist[:, 1], 'b-o', markersize=3, linewidth=1.2,
             label='Robot trajectory')
    ax1.plot(x_hist[0, 0], x_hist[0, 1], 'go', markersize=10,
             label='Start')
    ax1.plot(TARGET[0], TARGET[1], 'r*', markersize=15,
             label=f'Target ({TARGET[0]}, {TARGET[1]})')

    # 每隔一段画小箭头表示朝向
    arrow_step = max(1, len(x_hist) // 15)
    for i in range(0, len(x_hist), arrow_step):
        ax1.arrow(x_hist[i, 0], x_hist[i, 1],
                  0.02 * np.cos(x_hist[i, 2]),
                  0.02 * np.sin(x_hist[i, 2]),
                  head_width=0.03, head_length=0.04, fc='blue', ec='blue', alpha=0.6)

    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Differential Drive NMPC — XY Trajectory (Bird View)')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    # ---- 图2: 状态随时间变化 ----
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels_state = ['X [m]', 'Y [m]', 'Theta [rad]']
    colors_state = ['C0', 'C1', 'C2']
    for i, (ax, label, color) in enumerate(zip(axes2, labels_state, colors_state)):
        ax.plot(t_sim, x_hist[:, i], color=color, linewidth=1.2)
        ax.axhline(y=TARGET[i], color='red', linestyle='--', alpha=0.5,
                   label=f'Target ({TARGET[i]:.1f})')
        ax.set_ylabel(label)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes2[-1].set_xlabel('Time [s]')
    axes2[0].set_title('State Trajectories vs Time')
    fig2.tight_layout()

    # ---- 图3: 控制量随时间变化 ----
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t_u = t_sim[:-1]  # 控制有 M-1 个值

    ax = axes3[0]
    ax.step(t_u, u_hist[:-1, 0], where='post', color='C0', linewidth=1.2,
            label='v [m/s]')
    ax.axhline(y=U_MAX[0], color='red', linestyle=':', alpha=0.4,
               label=f'Bound (±{U_MAX[0]})')
    ax.axhline(y=U_MIN[0], color='red', linestyle=':', alpha=0.4)
    ax.set_ylabel('Linear Velocity v [m/s]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    ax.step(t_u, u_hist[:-1, 1], where='post', color='C1', linewidth=1.2,
            label='omega [rad/s]')
    ax.axhline(y=U_MAX[1], color='red', linestyle=':', alpha=0.4,
               label=f'Bound (±{U_MAX[1]})')
    ax.axhline(y=U_MIN[1], color='red', linestyle=':', alpha=0.4)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angular Velocity ω [rad/s]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    axes3[0].set_title('Control Inputs vs Time')
    fig3.tight_layout()

    # ---- 保存图片 ----
    out_dir = '../python/test_output'
    import os
    os.makedirs(out_dir, exist_ok=True)
    fig1.savefig(f'{out_dir}/trajectory_xy.png', dpi=150)
    fig2.savefig(f'{out_dir}/states_vs_time.png', dpi=150)
    fig3.savefig(f'{out_dir}/controls_vs_time.png', dpi=150)
    print(f"\n  Figures saved to: {os.path.abspath(out_dir)}/")

    plt.show()


def main():
    print("\n" + "=" * 65)
    print("  Differential Drive NMPC Test (CasADi IPOPT)")
    print("=" * 65)
    print(f"  Horizon: T={Tf}s, N={N}, dt={dt:.3f}s")
    print(f"  State weight: Q=diag({Q_DIAG})")
    print(f"  Control weight: R=diag({R_DIAG})")
    print(f"  Rate weight: R_rate=diag({R_RATE_DIAG})")
    print(f"  Control bounds: v in [{U_MIN[0]},{U_MAX[0]}], omega in [{U_MIN[1]},{U_MAX[1]}]")

    # 1. Build solver
    print("\n  Building MPC solver (IPOPT)...")
    mpc_solve = build_mpc_solver()
    print("  MPC solver built.")

    # 2. Run simulation
    x_hist, u_hist, cost_hist, time_hist = run_closed_loop_simulation(mpc_solve)

    # 3. Plot
    print("\n  Generating figures...")
    plot_results(x_hist, u_hist, cost_hist, time_hist)

    print("\n  Test completed.")


if __name__ == '__main__':
    main()