import os

CODE = os.path.dirname(os.path.abspath(__file__))
Q2 = os.path.dirname(CODE)
RESULT = os.path.join(Q2, 'result')
IMAGE = os.path.join(Q2, 'image')
INPUT = os.path.join(Q2, 'input')

def _data_root():
    env = os.environ.get('BDC_RESOURCES')
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    p = Q2
    for _ in range(6):
        p = os.path.dirname(p)
        for name in ('resources', 'Resources'):
            cand = os.path.join(p, name)
            if os.path.isdir(cand):
                return cand
    return os.path.abspath(os.path.join(Q2, '..', '..', 'resources'))

RESOURCES = _data_root()
