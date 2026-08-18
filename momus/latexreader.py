#-----------------------------------------------------------------------#
# momus.latexreader v0.1.0
# By Hunter Brooks, at UToledo
#-----------------------------------------------------------------------#

# Import Packages
#-----------------------------------------------------------------------#
import re
import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex
#-----------------------------------------------------------------------#



# Sanitize LaTeX for LIATE Compliance
#-----------------------------------------------------------------------#
def _sanitize_latex(latex_eq):
    latex_eq = re.sub(r'\\sin\s*\^{-?1}', r'\\arcsin', latex_eq)
    latex_eq = re.sub(r'\\cos\s*\^{-?1}', r'\\arccos', latex_eq)
    latex_eq = re.sub(r'\\tan\s*\^{-?1}', r'\\arctan', latex_eq)
    latex_eq = re.sub(r'\\csc\s*\^{-?1}', r'\\arccsc', latex_eq)
    latex_eq = re.sub(r'\\sec\s*\^{-?1}', r'\\arcsec', latex_eq)
    latex_eq = re.sub(r'\\cot\s*\^{-?1}', r'\\arccot', latex_eq)
    return latex_eq
#-----------------------------------------------------------------------#



# Read LaTeX Equation
#-----------------------------------------------------------------------#
def read_latex_equation(latex_eq, default_x="x", backend="lark"):
    
    # Preprocess string
    clean_eq = _sanitize_latex(latex_eq)

    # Parse LaTeX Equation
    try:
        expr = parse_latex(clean_eq, backend=backend)
    except Exception:
        expr = parse_latex(clean_eq, backend="antlr")
    
    # Convert any accidental Dummy objects into standard Symbols
    dummy_replacements = {
        sub: sp.Symbol(sub.name) 
        for sub in expr.free_symbols 
        if isinstance(sub, sp.Dummy)
    }
    if dummy_replacements:
        expr = expr.subs(dummy_replacements)

    # Mathematical Constants Mapping (LIATE Protection)
    mathematical_constants = {
        sp.pi,
        sp.E,
        sp.I,
        sp.oo,
        sp.Symbol("pi"),
        sp.Symbol("e"),
        sp.Symbol("E"),
        sp.Symbol("i"),
        sp.Symbol("j"),
    }

    # Replace/Normalize Constants so they aren't treated as Parameters
    expr = expr.subs({
        sp.pi: sp.Float(np.pi),
        sp.Symbol("pi"): sp.Float(np.pi),
        sp.Symbol("\\pi"): sp.Float(np.pi),
        sp.E: sp.E,
        sp.Symbol("e"): sp.E,
        sp.Symbol("E"): sp.E,
        sp.Symbol("\\mathrm{e}"): sp.E,
        sp.Symbol("i"): sp.I,
        sp.Symbol("j"): sp.I,
    })

    # Normalize Logarithms: Convert log(x, E) to natural log log(x)
    expr = expr.replace(lambda node: isinstance(node, sp.log) and len(node.args) == 2 and node.args[1] == sp.E, lambda node: sp.log(node.args[0]))

    # Find All Symbols
    symbols = sorted(expr.free_symbols, key=lambda symbol: str(symbol))

    # Find Independent Variable
    x_symbol = next((symbol for symbol in symbols if str(symbol) == default_x), None)
    if x_symbol is None and symbols:
        x_symbol = symbols[0]
    if x_symbol is None:
        raise ValueError(
            "Could not find an independent variable "
            "in the LaTeX equation."
        )

    # Find Parameters (Excluding Constants and Independent Variable)
    coefficients = [symbol for symbol in symbols if (symbol != x_symbol and symbol not in mathematical_constants and str(symbol) not in {"pi", "e", "E", "i", "j", "\\pi"})]

    # Sort Parameters
    def coefficient_sort_key(symbol):
        name = str(symbol)
        if name.startswith("a_") and name[2:].isdigit():
            return (0, int(name[2:]))
        return (1, name)
    coefficients = sorted(coefficients, key=coefficient_sort_key)
    
    domain = sp.calculus.util.continuous_domain(expr, x_symbol, sp.S.Reals)
    is_continuous_everywhere = domain == sp.S.Reals

    # Number of Coefficients
    num_coeffs = len(coefficients)
    model_func = sp.lambdify([x_symbol, *coefficients], expr, modules=["numpy"])
    return (expr, x_symbol, coefficients, num_coeffs, model_func, [is_continuous_everywhere, domain])
#-----------------------------------------------------------------------#