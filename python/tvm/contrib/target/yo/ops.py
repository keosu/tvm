
"""supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by one target.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to Yo.
"""
import logging
from functools import reduce

import tvm.ir
from tvm.ir import Op
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr import GlobalVar
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.expr import const

from tvm.relay.analysis import analysis as _analysis
from tvm.relay import expr as _expr

from tvm.relay.expr import Call, TupleGetItem
from tvm.relay import _ffi_api
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant, is_expr, rewrite, DFPatternCallback
from tvm.relay.op.contrib.register  import register_pattern_table


logger = logging.getLogger("Yo")
supported_post_elts = ["nn.relu", "tanh", "sigmoid", "clip", "gelu", "swish", "mish", None]


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by Yo.

    Parameters
    ----------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by Yo.
    """

    @tvm.ir.register_op_attr(op_name, "target.yo")
    def _func_wrapper(expr):
        args = expr.args
        if any([x.checked_type.dtype == "int64" for x in args]):
            logger.info("Yo does not support int64.")
            return False
        # Yo does not support pooling with ceil_mode = True.
        if "pool" in op_name:
            attrs = dict(get_attrs(expr))
            if "ceil_mode" in attrs.keys() and attrs["ceil_mode"]:
                return False
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv1d")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.conv3d")
_register_external_op_helper("nn.conv2d_transpose")
_register_external_op_helper("nn.conv3d_transpose")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.global_avg_pool2d")
_register_external_op_helper("nn.max_pool3d")
_register_external_op_helper("nn.avg_pool3d")
_register_external_op_helper("abs")
_register_external_op_helper("clip")
_register_external_op_helper("exp")
_register_external_op_helper("log")
_register_external_op_helper("sqrt")
_register_external_op_helper("round")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("tanh")
_register_external_op_helper("sigmoid")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("add")
_register_external_op_helper("multiply")
_register_external_op_helper("nn.layer_norm")
_register_external_op_helper("nn.batch_matmul")
_register_external_op_helper("qnn.add")

 

def make_qnn_conv2d_pattern():
    """Make qnn.conv2d based pattern supported by Yo

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    data = wildcard()
    weight = is_constant()
    bias = is_constant()
    o_scl = is_constant()
    dst_zp = is_constant()
    act_scl = is_constant()
    sum_scl = is_constant()
    sum_src = wildcard()

    zero_zp = is_expr(const(0, dtype="int32"))

    pat = is_op("qnn.conv2d")(data, weight, zero_zp, zero_zp, is_constant(), is_constant())
    pat = is_op("cast")(pat)
    pat = is_op("add")(pat, bias) | pat  # optional bias
    pat = is_op("multiply")(pat, o_scl)
    pat = is_op("clip")(pat)  # TBD, not only clip
    pat = is_op("multiply")(pat, act_scl) | pat  # optional multiply. Ex: act_scl == 1
    pat = is_op("add")(pat, sum_scl * is_op("cast")(sum_src)) | pat  # optional sum
    pat = is_op("add")(pat, dst_zp) | pat  # optional dst_zp, can be dst_zp == 0
    pat = is_op("cast")(pat)

    return "dnnl.qnn.conv2d", pat


def make_qnn_dense_pattern():
    """Make qnn.dense based pattern supported by Yo

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    data = wildcard()
    weight = is_constant()
    bias = is_constant()
    o_scl = is_constant()
    dst_zp = is_constant()
    act_scl = is_constant()
    sum_scl = is_constant()
    sum_src = wildcard()

    zero_zp = is_expr(const(0, dtype="int32"))

    pat = is_op("qnn.dense")(data, weight, zero_zp, zero_zp, is_constant(), is_constant())
    pat = is_op("cast")(pat)
    pat = is_op("add")(pat, bias) | pat  # optional bias
    pat = is_op("multiply")(pat, o_scl)
    pat = is_op("clip")(pat)  # TBD, not only clip
    pat = is_op("multiply")(pat, act_scl) | pat  # optional multiply. ex act_scl == 1
    pat = is_op("add")(pat, sum_scl * is_op("cast")(sum_src)) | pat  # optional sum
    pat = is_op("add")(pat, dst_zp) | pat  # optional dst_zp, can be dst_zp == 0
    pat = is_op("cast")(pat)

    return "dnnl.qnn.dense", pat


def make_qnn_add_pattern(): 
    a = wildcard()
    b = wildcard() 

    pat = is_op("qnn.add")(a, b, is_constant(), is_constant(), is_constant(), is_constant(),is_constant(), is_constant()) 

    return "qnn.add", pat



@register_pattern_table("yo")
def pattern_table():
    """Create dnnl patterns.

    Returns
    -------
    op_patterns : List[op_pattern]
        Created patterns.
    """
    op_patterns = list()
    op_patterns.append(make_qnn_conv2d_pattern())
    op_patterns.append(make_qnn_dense_pattern()) 
    # op_patterns.append(make_qnn_add_pattern()) 
    return op_patterns

 