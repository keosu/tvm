import tvm

from tvm import te
from tvm import autotvm
from tvm import topi
from tvm import tir
from tvm.tir.op import exp
import tvm.relay.op.op as reg

from tvm.relay.op import strategy 
from tvm.relay.op.op import OpPattern, OpStrategy

from functools import reduce

@tvm.ir.register_op_attr("qnn.add", "target.yodev")
def add(attrs, *args):
    return True


#=============================
SCHED_TABLE = {}


def register_yo_schedule(op_tag):
    def _decorate(schedule_function):
        SCHED_TABLE[op_tag] = schedule_function

        def wrapper(op, s, parent_ops, scheduled):
            if op.tag in SCHED_TABLE:
                return SCHED_TABLE[op.tag](op, s, parent_ops, scheduled)
            return s

        return wrapper

    return _decorate

def compute_qnn_add_yo(*inputs):
    a, b = inputs[0], inputs[1]
    para = [inputs[i].op.body[0].value for i in range(2, 8)]

    size_a = reduce(lambda i, j: i * j, list(a.shape))
    size_b = reduce(lambda i, j: i * j, list(b.shape))
    if size_a != size_b:
        raise ValueError(f"Input a and b have different buffer size: {size_a} vs {size_b}")

    def _identity(input, name, tag="", attr=None):
        ret = lambda *i: input(*i)
        return te.compute(input.shape, ret, name=name, tag=tag, attrs=attr)

    def _concat(a, b, name):
        ret = lambda i, j, k: te.if_then_else(
            i == 0, a(*_get_nd_idx(k, a.shape)), b(*_get_nd_idx(k, b.shape))
        )
        return te.compute([2, 1, size_a], ret, name=name)

    def _split_add(input, name):
        ret = lambda *i: input(0, *i) + input(1, *i)
        return te.compute(input.shape[1:], ret, name=name)

    in_shared = _concat(a, b, "in_shared")
    in_local = _identity(in_shared, "in_local")

    # AIE computing func defination: just use a fake func
    out_local = _split_add(in_local, name="qnn.add")

    out_shared = _identity(out_local, "out_shared")
    out = _identity(out_shared, "out_global", tag="elementwise_add", attr={"para": para})

    return out

def schedule_binary_elementwise(op, s):
    intrin_name = op.tag

    Out = op.output(0)
    out_shared = s[Out].op.input_tensors[0]
    out_local = s[out_shared].op.input_tensors[0]
    in_local = s[out_local].op.input_tensors[0]
    in_shared = s[in_local].op.input_tensors[0]
    in_a, in_b = s[in_shared].op.input_tensors

    size = reduce(lambda i, j: i * j, list(in_a.shape)) 

    s[in_a].set_scope("global")
    s[in_b].set_scope("global")
    s[in_shared].set_scope("shared")
    s[in_local].set_scope("local")
    s[out_local].set_scope("local")
    s[out_shared].set_scope("shared")
    s[Out].set_scope("global")

    _, tOut_yo, tOut_xi, tOut_yi = s[Out].tile(
        Out.op.axis[0], Out.op.axis[1], x_factor=1, y_factor=12
    )

    s[in_a].compute_at(s[Out], tOut_yo)
    s[in_b].compute_at(s[Out], tOut_yo)
    s[in_shared].compute_at(s[Out], tOut_yo)
    s[in_local].compute_at(s[Out], tOut_yo)
    s[out_local].compute_at(s[Out], tOut_yo)
    s[out_shared].compute_at(s[Out], tOut_yo)

    s[in_shared].pragma(in_shared.op.axis[1], "pragma_b")
    s[in_shared].pragma(in_shared.op.axis[1], "pragma_a")
    s[in_shared].unroll(in_shared.op.axis[0])

    s[in_local].pragma(in_local.op.axis[0], "pragma_a")
 
    kernel_pragma = "my_intrin_{}*{}*{}*{}".format(
        intrin_name, 0, 1, 2
    )
    s[out_local].pragma(out_local.op.axis[0], kernel_pragma)

    s[out_shared].vectorize(out_shared.op.axis[1])
    s[out_shared].pragma(out_shared.op.axis[0], "pragma_b")

    s[Out].vectorize(tOut_yi)
    s[Out].pragma(tOut_xi, "pragma_a")
    s[Out].pragma(tOut_xi, "pragma_b")
    s[Out].pragma(tOut_yo, "aie_lock_barrier")
 
    return


# register_yo_schedule("qnn.add")(schedule_binary_elementwise) 


def schedule_qnn_add_yo(outs):
    assert len(outs) == 1
    s = te.create_schedule([x.op for x in outs])
    schedule_binary_elementwise(outs[0].op, s)
    return s


@tvm.target.override_native_generic_func("qnn_add_strategy_yo")
def qnn_add_strategy_yo(attrs, inputs, out_type, target): 
    op_stra = OpStrategy()
    op_stra.add_implementation(
        strategy.wrap_topi_compute(compute_qnn_add_yo),
        strategy.wrap_topi_schedule(schedule_qnn_add_yo),
        name="qnn.add.yo",
        plevel=100,
    )
    return op_stra

 

reg.register_strategy("qnn.add", qnn_add_strategy_yo)
reg.register_pattern("qnn.add", OpPattern.OUT_ELEMWISE_FUSABLE)