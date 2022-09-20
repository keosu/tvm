
import inspect
import os
import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_executor

# import tvm.relay.op.contrib.yo
# from tvm.relay.op.contrib.yo import partition_for_aie
from tvm.contrib.target import yo

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


pass_lst = []


@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_after_pass(self, mod, info):
        # print(f"Running pass: {info} ")
        # print("="*30)
        pass_lst.append(info.name)
        # print(mod)


def test_add(target_desc="default tgt"):
    print(f"in func:: {inspect.stack()[1].function  }")

    def _get_func(opname, shape_a, shape_b, shift_l, shift_r, dtype):
        a = relay.var("a", shape=shape_a, dtype=dtype)
        b = relay.var("b", shape=shape_b, dtype=dtype)

        op = relay.qnn.op.add if opname == "qnn.add" else relay.qnn.op.mul

        scale_in = relay.const(2 ** shift_l, "float32")
        scale_out = relay.const(2 ** shift_r, "float32")
        zero = relay.const(0, "int32")

        res = op(a, b, scale_in, zero, scale_in, zero, scale_out, zero)

        return relay.Function([a, b], res)

    def _compile(func, target="yo"):
        mod = tvm.IRModule()
        mod["main"] = func

        if target == "cpu":
            with tvm.transform.PassContext(3, instruments=[PrintIR()]):
                lib = relay.build(mod, "llvm", params={})
                return lib
        else:
            passes = [
                transform.MergeComposite(
                    yo.ops.pattern_table()),
                transform.AnnotateTarget("yo"),
                transform.MergeCompilerRegions(),
                transform.PartitionGraph(),
            ]

            with yo.target.YoTarget("yo"):
                cfg = {
                    "relay.ext.yo.options.target": f"{target_desc}"
                }
                with tvm.transform.PassContext(opt_level=3, config=cfg):
                    print(mod)
                    for p in passes:
                        mod = p(mod)
                        print(mod)
                    lib = relay.build(mod, "llvm", params={})
                    return lib

    def _run(lib, a, b):
        ctx = tvm.cpu(0)
        m = graph_executor.GraphModule(lib["default"](ctx))
        m.set_input("a", a)
        m.set_input("b", b)
        m.run()
        return m.get_output(0).numpy()

    def _verify(op_name, shape_a, shape_b, shift_l, shift_r, dtype):
        a = np.random.randint(-128, 127, shape_a, dtype=dtype)
        b = np.random.randint(-128, 127, shape_b, dtype=dtype)

        func = _get_func(op_name, shape_a, shape_b, shift_l, shift_r, dtype)

        libaie = _compile(func, "yo")
        libcpu = _compile(func, "cpu")

        cpuout = _run(libcpu, a, b)
        aieout = _run(libaie, a, b)

        # np.testing.assert_allclose(cpuout, aieout, rtol=1e-5, atol=1)

    _verify("qnn.add", (1, 64), (1, 64), 0, 0, "int16")


if __name__ == "__main__":
    test_add()
    print(f"pass count: {len(pass_lst)}")
