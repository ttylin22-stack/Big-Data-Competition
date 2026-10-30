import os
#以code文件夹为基点，创建路径
CODE = os.path.dirname(os.path.abspath(__file__))
#Q3总目录的相对路径
Q3 = os.path.dirname(CODE)
#依次创建剩下几个目录的路径
RESULT = os.path.join(Q3, 'result')
IMAGE = os.path.join(Q3, 'image')
INPUT = os.path.join(Q3, 'input')
#resources原始数据路径
RESOURCES = os.path.abspath(os.path.join(Q3, '..', '..', 'resources'))