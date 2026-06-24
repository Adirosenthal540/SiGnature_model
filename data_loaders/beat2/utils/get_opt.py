import os
from argparse import Namespace
import re
from os.path import join as pjoin


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip("-").lstrip("+")  # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r"^[-+]?[0-9]+\.[0-9]+$")
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip("-").lstrip("+")  # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag
