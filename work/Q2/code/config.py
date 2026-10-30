# -*- coding: utf-8 -*-
import os

CODE = os.path.dirname(os.path.abspath(__file__))   # .../work/Q2/code
Q2 = os.path.dirname(CODE)                          # .../work/Q2

RESULT = os.path.join(Q2, 'result')
IMAGE = os.path.join(Q2, 'image')
INPUT = os.path.join(Q2, 'input')


def _is_data_root(path):
    if not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(n.startswith('DataSet_1') or n.startswith('（1）') or n.startswith('(1)')
               for n in names)


def find_resources():
    for key in ('BDC_RESOURCES', 'GYIRONG_BASE'):
        env = os.environ.get(key)
        if env and _is_data_root(env):
            return os.path.abspath(env)

    p = Q2
    for _ in range(8):
        p = os.path.dirname(p)
        for cand in ('resources', 'Resources', '附件'):
            c = os.path.join(p, cand)
            if _is_data_root(c):
                return c

    # 兜底：Big_Data_Competition/resources
    return os.path.abspath(os.path.join(Q2, '..', '..', 'resources'))


RESOURCES = find_resources()
