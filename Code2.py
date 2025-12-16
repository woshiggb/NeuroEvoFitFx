import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication, convert_xor,
    split_symbols, auto_symbol
)
import numpy as np
from typing import Any, Dict, List, Union

class FullSymPyEvaluator:
    def __init__(self):
        self._init_symbols()
        self._init_functions()
        self._init_constants()
        
    def _init_symbols(self):
        self.symbols = {
            'x': sp.symbols('x'), 'y': sp.symbols('y'),
            'z': sp.symbols('z'), 't': sp.symbols('t'),
            'n': sp.symbols('n'), 'm': sp.symbols('m'),
            'k': sp.symbols('k'), 'a': sp.symbols('a'),
            'b': sp.symbols('b'), 'c': sp.symbols('c'),
        }
    
    def _init_functions(self):
        self.functions = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'cot': sp.cot, 'sec': sp.sec, 'csc': sp.csc,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'acot': sp.acot, 'asec': sp.asec, 'acsc': sp.acsc,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'coth': sp.coth, 'sech': sp.sech, 'csch': sp.csch,
            'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
            'acoth': sp.acoth, 'asech': sp.asech, 'acsch': sp.acsch,
            'exp': sp.exp, 'ln': sp.ln, 'log': sp.log,
            'log10': lambda x: sp.log(x, 10), 'log2': lambda x: sp.log(x, 2),
            'gamma': sp.gamma, 'beta': sp.beta, 'zeta': sp.zeta,
            'erf': sp.erf, 'erfc': sp.erfc, 'Ei': sp.Ei,
            'Si': sp.Si, 'Ci': sp.Ci, 'Li': sp.Li,
            'airyai': sp.airyai, 'airybi': sp.airybi,
            'besselj': sp.besselj, 'bessely': sp.bessely,
            'besseli': sp.besseli, 'besselk': sp.besselk,
            'factorial': sp.factorial, 'fibonacci': sp.fibonacci,
            'lucas': lambda n: sp.fibonacci(n-1) + sp.fibonacci(n+1),
            'sqrt': sp.sqrt, 'cbrt': lambda x: x**(sp.Rational(1, 3)),
            'abs': sp.Abs, 'sign': sp.sign, 'floor': sp.floor,
            'ceiling': sp.ceiling, 're': sp.re, 'im': sp.im,
            'conjugate': sp.conjugate, 'arg': sp.arg,
        }
    
    def _init_constants(self):
        self.constants = {
            'pi': sp.pi, 'E': sp.E, 'I': sp.I,
            'oo': sp.oo, '-oo': -sp.oo, 'zoo': sp.zoo,
            'nan': sp.nan, 'EulerGamma': sp.EulerGamma,
            'Catalan': sp.Catalan, 'GoldenRatio': sp.GoldenRatio,
            'TribonacciConstant': 1.839286755214161,
        }
    
    def parse(self, expr_str: str) -> sp.Expr:
        transformations = (
            standard_transformations +
            (implicit_multiplication, convert_xor, split_symbols, auto_symbol)
        )
        local_dict = {**self.symbols, **self.functions, **self.constants}
        return parse_expr(expr_str, transformations=transformations, local_dict=local_dict, evaluate=True)
    
    def eval(self, expr_str: str, **kwargs) -> Any:
        expr = self.parse(expr_str)
        if kwargs:
            subs_dict = {}
            for key, value in kwargs.items():
                if key in self.symbols:
                    subs_dict[self.symbols[key]] = value
                else:
                    subs_dict[sp.symbols(key)] = value
            expr = expr.subs(subs_dict)
        try:
            if expr.is_number:
                return expr.evalf()
            else:
                return sp.simplify(expr)
        except:
            return expr
    
    def numeric_eval(self, expr_str: str, **kwargs) -> float:
        result = self.eval(expr_str, **kwargs)
        if isinstance(result, sp.Expr):
            if result.is_number:
                return float(result.evalf())
            else:
                raise ValueError("表达式包含未定义的符号")
        return float(result)
    
    def solve(self, equations: Union[str, List[str]], **kwargs) -> Dict:
        if isinstance(equations, str):
            equations = [equations]
        eq_exprs = []
        for eq in equations:
            if '=' in eq:
                lhs, rhs = eq.split('=', 1)
                expr = self.parse(lhs) - self.parse(rhs)
            else:
                expr = self.parse(eq)
            eq_exprs.append(expr)
        symbols = set()
        for expr in eq_exprs:
            symbols.update(expr.free_symbols)
        solutions = sp.solve(eq_exprs, list(symbols), **kwargs)
        if isinstance(solutions, list):
            result = []
            for sol in solutions:
                if isinstance(sol, dict):
                    result.append(sol)
                elif isinstance(sol, tuple):
                    result.append(dict(zip(symbols, sol)))
            return result
        elif isinstance(solutions, dict):
            return [solutions]
        else:
            return solutions
    
    def diff(self, expr_str: str, var: str = 'x', n: int = 1) -> sp.Expr:
        expr = self.parse(expr_str)
        var_sym = sp.symbols(var)
        return sp.diff(expr, var_sym, n)
    
    def integrate(self, expr_str: str, var: str = 'x', **kwargs) -> sp.Expr:
        expr = self.parse(expr_str)
        var_sym = sp.symbols(var)
        if 'limits' in kwargs:
            return sp.integrate(expr, kwargs['limits'])
        else:
            return sp.integrate(expr, var_sym)
    
    def limit(self, expr_str: str, var: str = 'x', point: Any = 0, **kwargs) -> sp.Expr:
        expr = self.parse(expr_str)
        var_sym = sp.symbols(var)
        return sp.limit(expr, var_sym, point, **kwargs)
    
    def series(self, expr_str: str, var: str = 'x', point: Any = 0, n: int = 6) -> sp.Expr:
        expr = self.parse(expr_str)
        var_sym = sp.symbols(var)
        return sp.series(expr, var_sym, point, n).removeO()
    
    def matrix(self, matrix_str: str) -> sp.Matrix:
        if matrix_str.startswith('[') and matrix_str.endswith(']'):
            return sp.Matrix(sp.sympify(matrix_str))
        else:
            expr = self.parse(matrix_str)
            if isinstance(expr, sp.Matrix):
                return expr
            else:
                return sp.Matrix([expr])
    
    def latex(self, expr_str: str) -> str:
        expr = self.parse(expr_str)
        return sp.latex(expr)
    
    def python_code(self, expr_str: str) -> str:
        expr = self.parse(expr_str)
        return sp.pycode(expr)
    
    def plot(self, expr_str: str, **kwargs):
        import matplotlib.pyplot as plt
        expr = self.parse(expr_str)
        if 'x' in str(expr.free_symbols):
            sp.plotting.plot(expr, **kwargs)
        elif len(expr.free_symbols) == 2:
            vars_list = list(expr.free_symbols)
            sp.plotting.plot3d(expr, **{str(vars_list[0]): kwargs.get('xrange', (-5, 5)),
                                        str(vars_list[1]): kwargs.get('yrange', (-5, 5))})
    
    def define_symbol(self, name: str):
        self.symbols[name] = sp.symbols(name)
        return self.symbols[name]
    
    def define_function(self, name: str, func_str: str):
        expr = self.parse(func_str)
        self.functions[name] = lambda *args: expr.subs(
            {sp.symbols(f'x{i}'): arg for i, arg in enumerate(args, 1)}
        ) if 'x' in func_str else expr
    
    def lambdify(self, expr_str: str, vars: List[str] = None):
        expr = self.parse(expr_str)
        if vars is None:
            vars = [str(sym) for sym in expr.free_symbols]
        return sp.lambdify(vars, expr, modules=['numpy', 'sympy'])




