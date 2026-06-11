"""
差速驱动底盘运动学模型 — 用于 acados NMPC 求解器
=====================================================

模型描述:
    标准差速驱动 (differential drive) 机器人运动学模型。
    机器人由左右两轮差速控制，通过线速度 v 和角速度 ω 实现平面运动。

状态变量 (3 维):
    x     : 机器人质心在世界坐标系下的 x 坐标 [m]
    y     : 机器人质心在世界坐标系下的 y 坐标 [m]
    theta : 机器人朝向角 [rad]，即车头方向与 x 轴夹角

控制输入 (2 维):
    v     : 线速度 [m/s]，沿车头方向为正
    omega : 角速度 [rad/s]，逆时针为正

连续时间动力学 (unicycle model):
    dx/dt     = v * cos(theta)
    dy/dt     = v * sin(theta)
    dtheta/dt = omega

注意:
    - 这是一个纯运动学模型，不考虑质量、惯量、轮滑等动力学因素
    - 实际机器人中 v 和 omega 由左右轮转速换算得到，这里直接作为控制输入
    - 该模型在低速 (≤1.5 m/s) 下对大多数差速底盘有较好的近似精度
"""

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos


def export_differential_drive_model() -> AcadosModel:
    """
    导出差速驱动底盘运动学模型，返回 acados 可用的 AcadosModel 对象。

    Returns
    -------
    model : AcadosModel
        包含状态 x, 状态导数 xdot, 控制 u, 显式动力学 f_expl_expr 的模型结构。
    """
    model_name = 'differential_drive'

    # -----------------------------------------------------------
    # 1. 定义符号变量：状态向量 x = [x, y, theta]
    # -----------------------------------------------------------
    x     = SX.sym('x')           # x 坐标 [m]
    y     = SX.sym('y')           # y 坐标 [m]
    theta = SX.sym('theta')       # 朝向角 [rad]
    x_sym = vertcat(x, y, theta)  # 拼成 3×1 列向量

    # 状态导数符号 (acados 需要显式声明，即使不在表达式中直接使用)
    x_dot     = SX.sym('x_dot')
    y_dot     = SX.sym('y_dot')
    theta_dot = SX.sym('theta_dot')
    xdot      = vertcat(x_dot, y_dot, theta_dot)

    # -----------------------------------------------------------
    # 2. 定义控制输入：u = [v, omega]
    # -----------------------------------------------------------
    v     = SX.sym('v')           # 线速度 [m/s]
    omega = SX.sym('omega')       # 角速度 [rad/s]
    u     = vertcat(v, omega)     # 拼成 2×1 列向量

    # -----------------------------------------------------------
    # 3. 显式动力学：f(x, u) = [v*cosθ, v*sinθ, ω]
    # -----------------------------------------------------------
    f_expl = vertcat(
        v * cos(theta),           # dx/dt
        v * sin(theta),           # dy/dt
        omega                     # dtheta/dt
    )

    # -----------------------------------------------------------
    # 4. 组装 AcadosModel 并返回
    # -----------------------------------------------------------
    model           = AcadosModel()
    model.name      = model_name
    model.x         = x_sym        # 状态向量 (3×1)
    model.xdot      = xdot         # 状态导数符号 (3×1)
    model.u         = u            # 控制向量 (2×1)
    model.f_expl_expr = f_expl     # 显式动力学表达式

    return model