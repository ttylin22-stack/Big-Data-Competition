# -*- coding: utf-8 -*-
"""交付包统一路径：相对本文件定位。"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
BASE = os.path.abspath(os.path.join(PKG, '..', '..', '附件'))
IN = os.path.join(PKG, '05_输入')
DATA = os.path.join(PKG, '03_结果')
FIGS = os.path.join(PKG, '04_图表')
