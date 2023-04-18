import cgp
import numpy as np

class sat_add(cgp.OperatorNode):
  """ A node that calculates the exponential of its input.
  """
  _arity = 2
  _def_output = "np.add(x_0, x_1)"
  _def_numpy_output = "np.add(x_0, x_1)"


class sat_sub(cgp.OperatorNode):
  """ A node that calculates the exponential of its input.
  """
  _arity = 2
  _def_output = "np.subtract(x_0, x_1)"
  _def_numpy_output = "np.subtract(x_0, x_1)"


class cgp_min(cgp.OperatorNode):
  """ A node that calculates the exponential of its input.
  """
  _arity = 2
  _def_output = "np.min((x_0, x_1), axis=0)"
  _def_numpy_output = "np.min((x_0, x_1), axis=0)"


class cgp_max(cgp.OperatorNode):
  """ A node that calculates the exponential of its input.
  """
  _arity = 2
  _def_output = "np.max((x_0, x_1), axis=0)"
  _def_numpy_output = "np.max((x_0, x_1), axis=0)"

class greater_than(cgp.OperatorNode):
  """ A node that calculates the exponential of its input.
  """
  _arity = 2
  _def_output = "np.where(x_0 > x_1, 1.0, -1.0)"
  _def_numpy_output = "np.where(x_0 > x_1, 1.0, -1.0)"

class sat_mul(cgp.OperatorNode):
  """ A node that calculates the exponential of its input.
  """
  _arity = 2
  _def_output = "x_0*x_1"
  _def_numpy_output = "x_0*x_1"


class const_random(cgp.ConstantFloat):
  _arity = 0
  _initial_values = {"<w>": lambda: np.random.rand()}
  _def_output = "<w>"
  _def_numpy_output = "<w>"


class scale_up(cgp.OperatorNode):
  """ A node that calculates the exponential of its input.
  """
  _arity = 1
  _def_output = "np.multiply(x_0, 2)"
  _def_numpy_output = "np.multiply(x_0, 2)"

class scale_down(cgp.OperatorNode):
  """ A node that calculates the exponential of its input.
  """
  _arity = 1
  _def_output = "np.divide(x_0, 2)"
  _def_numpy_output = "np.divide(x_0, 2)"

class negation(cgp.OperatorNode):
  """ A node that calculates the exponential of its input.
  """
  _arity = 1
  _def_output = "np.negative(x_0)"
  _def_numpy_output = "np.negative(x_0)"
