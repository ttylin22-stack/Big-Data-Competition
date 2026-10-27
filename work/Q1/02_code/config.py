# 本文件应使用UTF-8编码
"""交付包统一路径：全部相对本文件定位，换机器可直接运行。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))                  # 02_代码
PKG = os.path.dirname(HERE)                                         # 问题1_交付包
BASE = os.path.abspath(os.path.join(PKG, '..', '..', '附件'))       # 赛道B/附件（原始数据）
DATA = os.path.join(PKG, '03_结果')
FIGS = os.path.join(PKG, '04_图表')
TMP = os.path.join(PKG, '05_中间数据')
