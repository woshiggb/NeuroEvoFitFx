from Code2 import *
expreessions = [f"{input("Your Way:(Like:a**2 - b ** 3)")}]
for i, expr_str in enumerate(expressions, 1):
    try:
      func = create_2d_function(expr_str)
      print(f"\n{i}. 表达式: {expr_str}")
      print(f"   检测到的变量: {func.variables}")
      print(f"   测试 f(1, 2) = {func(1, 2):.4f}")
    except Exception as e:
      print(f"\n{i}. 错误: {e}")
          
F = create_2d_function(expr_str, var_names=['a', 'b'])
from Code import *

