import os
#以code文件夹为基点，创建路径
CODE = os.path.dirname(os.path.abspath(__file__))
#Q4总目录的相对路径
Q4 = os.path.dirname(CODE)
#依次创建剩下几个目录的路径
RESULT = os.path.join(Q4, 'result')
IMAGE = os.path.join(Q4, 'image')
INPUT = os.path.join(Q4, 'input')
#resources原始数据路径
RESOURCES = os.path.abspath(os.path.join(Q4, '..', '..', 'resources'))
