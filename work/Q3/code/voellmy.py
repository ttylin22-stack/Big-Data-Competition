# -*- coding: utf-8 -*-
"""运移模型工具：Voellmy 一维（正确的阻力项 a = g sinθ - μg cosθ - g v²/(ξh)）"""
import numpy as np

G = 9.81

def run_voellmy(z, s_km, mu=0.05, xi=1000.0, hflow=8.0, v0=15.0, ds=20.0, vmin=1.0, smooth_m=600.0):
    """沿沟道积分，返回 (时间数组, 速度数组, 是否到达终点)

    smooth_m: 对高程剖面做移动平均的窗口长度（米）。像素级路径有台阶状锯齿，
              不平滑会让坡度在"陡/平"之间跳动，速度曲线出现假振荡。
    """
    s_m = s_km * 1000
    z = np.asarray(z, float)
    s_m = np.asarray(s_m, float)
    # 重采样到 ds 步长
    grid = np.arange(s_m[0], s_m[-1], ds)
    zz = np.interp(grid, s_m, z)
    # 平滑：窗口至少 3 个点，且不超过剖面长度
    win = int(max(3, min(round(smooth_m / ds), len(zz) // 2)))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    zz = np.convolve(np.pad(zz, win // 2, mode='edge'), kernel, mode='valid')
    v = v0
    t = 0.0
    ts, vs, ss = [0.0], [v], [0.0]
    for k in range(1, len(grid)):
        dsi = grid[k] - grid[k - 1]
        slope = max((zz[k - 1] - zz[k]) / dsi, 0.0)
        theta = np.arctan(slope)
        a = G * np.sin(theta) - mu * G * np.cos(theta) - G * v * v / (xi * hflow)
        v2 = v * v + 2 * a * dsi
        if v2 <= vmin ** 2:
            return np.array(ts), np.array(vs), False
        v = np.sqrt(v2)
        t += dsi / (0.5 * (v + np.sqrt(max(v2 - 2 * a * dsi, 0))))
        ts.append(t); vs.append(v); ss.append(grid[k])
    return np.array(ts), np.array(vs), True

def calibrate_mu(z, s_km, xi=1000.0, hflow=8.0, v0=15.0):
    """二分搜索 μ，使流动恰好到达终点（runout 校准）"""
    lo, hi = 0.005, 0.30
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        _, _, ok = run_voellmy(z, s_km, mu=mid, xi=xi, hflow=hflow, v0=v0)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo
