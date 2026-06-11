#!/usr/bin/env python3
"""
差速驱动底盘 NMPC 求解器代码生成脚本
======================================

功能:
    基于 diff_model.py 中定义的差速底盘运动学模型，配置一个非线性 MPC (NMPC)
    最优控制问题 (OCP)，然后调用 acados_template 生成 C 语言求解器代码。

工作流程:
    1. 从 diff_model 导入差速驱动运动学模型
    2. 创建 AcadosOcp 对象并配置:
       - 预测时域: N=30 步, Tf=1.0s (即控制频率 30 Hz)
       - 代价函数: 非线性最小二乘 (NONLINEAR_LS)
         * 路径代价: 状态误差 + 控制量 + 控制变化率
         * 终端代价: 终端状态误差
       - 约束: 状态边界、控制量边界、初始状态
    3. 设置求解器选项 (SQP_RTI + HPIPM + GAUSS_NEWTON)
    4. 生成 C 代码到 ../acados_generated/ 目录

生成的 C 代码用途:
    编译进 learning_mpc C++ 项目，作为在线 MPC 求解器的核心。

依赖:
    - acados_template (pip install acados_template)
    - casadi, numpy (由 acados_template 自动安装)
"""

import numpy as np
import casadi as ca

from acados_template import AcadosOcp, AcadosOcpSolver
from diff_model import export_differential_drive_model


def create_ocp_solver() -> AcadosOcpSolver:
    """
    配置并生成差速驱动底盘的 NMPC 求解器。

    Returns
    -------
    ocp_solver : AcadosOcpSolver
        已配置好的 OCP 求解器实例，可直接用于闭环仿真。
    """
    # ============================================================
    # 1. 创建 OCP 对象并加载模型
    # ============================================================
    ocp   = AcadosOcp()
    model = export_differential_drive_model()
    ocp.model = model

    # ============================================================
    # 2. 问题维度定义
    # ============================================================
    Tf = 1.0              # 预测时域总长度 [s]
    nx = model.x.rows()   # 状态维度 = 3 (x, y, theta)
    nu = model.u.rows()   # 控制维度 = 2 (v, omega)
    N  = 30               # 离散化步数 → 控制间隔 = Tf/N = 0.033s ≈ 30Hz

    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = Tf

    # ============================================================
    # 3. 代价函数设计 (非线性最小二乘型)
    # ============================================================
    # 权重矩阵: 数值越大对该项的惩罚越重
    #   Q: 状态跟踪权重 — 希望状态快速收敛到参考值
    #   R: 控制量权重   — 希望控制量尽量小 (节能)
    #   R_rate: 控制变化率权重 — 希望控制量变化平缓 (防抖动)
    Q_mat      = np.diag([25.0, 25.0, 25.0])   # [x, y, theta] 跟踪精度
    R_mat      = np.diag([0.1, 0.1])            # [v, omega] 控制抑制
    R_rate_mat = np.diag([2.0, 2.0])            # [Δv, Δomega] 变化率抑制

    # ---- 前一时刻控制量作为参数 (用于惩罚控制变化率) ----
    u_prev               = ca.SX.sym('u_prev', nu)
    ocp.model.p          = u_prev
    ocp.parameter_values = np.zeros(nu)

    # ---- 路径代价 (在每个离散点上施加) ----
    # cost_y_expr: 残差向量 = [x - x_ref, y - y_ref, theta - theta_ref,
    #                          v, omega, v - v_prev, omega - omega_prev]
    ocp.cost.cost_type    = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u, model.u - u_prev)
    ocp.cost.yref         = np.zeros((nx + nu + nu,))   # 参考值全为零
    ocp.cost.W            = ca.diagcat(Q_mat, R_mat, R_rate_mat).full()

    # ---- 终端代价 (仅在最后一步施加) ----
    ocp.cost.cost_type_e    = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref_e         = np.zeros((nx,))
    ocp.cost.W_e            = Q_mat   # 终端与路径状态权重一致

    # ============================================================
    # 4. 约束定义
    # ============================================================
    # ---- 状态边界: x∈[-10,10]m, y∈[-10,10]m, theta∈[-π,π]rad ----
    ocp.constraints.lbx   = np.array([-10.0, -10.0, -np.pi])
    ocp.constraints.ubx   = np.array([10.0, 10.0, np.pi])
    ocp.constraints.idxbx = np.array([0, 1, 2])

    # ---- 控制量边界: v∈[-1.5, 1.5] m/s, omega∈[-1.5, 1.5] rad/s ----
    ocp.constraints.lbu   = np.array([-1.5, -1.5])
    ocp.constraints.ubu   = np.array([1.5, 1.5])
    ocp.constraints.idxbu = np.array([0, 1])

    # ---- 初始状态 (仿真时动态设置，这里先填 0) ----
    ocp.constraints.x0 = np.zeros(nx)

    # ============================================================
    # 5. 求解器选项
    # ============================================================
    ocp.solver_options.qp_solver       = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hpipm_mode      = 'SPEED'
    ocp.solver_options.hessian_approx  = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    # ============================================================
    # 6. 生成 C 代码
    # ============================================================
    ocp.code_export_directory = '../acados_generated'

    ocp_solver = AcadosOcpSolver(
        ocp,
        json_file='../acados_generated/acados_ocp.json'
    )

    # ---- 打印生成信息 ----
    print(f"[信息] 求解器生成成功: {model.name}")
    print(f"       状态维度  (nx) = {nx}")
    print(f"       控制维度  (nu) = {nu}")
    print(f"       预测步数  (N)  = {N}")
    print(f"       预测时域  (Tf) = {Tf} s")
    print(f"       控制频率        = {N/Tf:.1f} Hz")

    return ocp_solver


def main():
    """主入口: 生成求解器并输出摘要。"""
    solver = create_ocp_solver()
    return solver


if __name__ == '__main__':
    main()