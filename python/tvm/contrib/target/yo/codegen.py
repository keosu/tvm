"""Codegen"""

import json
import numpy as np
import warnings
import logging
import numbers
import struct
from functools import reduce

import tvm
import tvm._ffi

from tvm import relay, te, tir
from tvm.driver.build_module import get_binds 
 
from .transform import StmtVisitor, Evaluator

logger = logging.getLogger("yo")
 
class CodegenYo(tvm.relay.ExprVisitor): 
    def __init__(self, target, model_name, func):
        super(CodegenYo, self).__init__()
        self.model_name = model_name
        self.target = target
        self.func = func
        self.params = {} 
        self.tensors = {} 
        self.code = []

    def visit_prim_func(self, prim_func):
        return "code"
        # should vist the func and generate the code
        # return PrimFuncCodegen(prim_func).create()

    def lower(self):
        print("[relay funcs]: \n", self.func) 
        func = self.func
        
        tc = relay.backend.te_compiler.get()
        # GetYoTarget
        from .target import GetYoTarget
        tgt = GetYoTarget() 
        s = tc.lower(func, tgt)

        print("[cached funcs]: \n", s.funcs.functions)
        return "dummy code"

        # TODO generate code
        # with build_config(str(self.target), opt_level=3, debug_flag=3): 
        #     code = self.visit_prim_func(prim_func)
        #     return code


@tvm._ffi.register_func("relay.ext.yo")
def yo_codegen(func):
    print("======= in py Yo-codegen")
    assert isinstance(func, tvm.relay.function.Function)
    # env = get_env()
    name = str(func.attrs.global_symbol)
    
    mod = tvm.IRModule()
    mod["main"] = func
    mod["main"] = relay.transform.Defunctionalization(mod["main"], mod)
    mod = relay.transform.InferType()(mod)
    func = mod["main"]

    codegen = CodegenYo("target", name, func)
    code = codegen.lower()  

    fcreate = tvm._ffi.get_global_func("tvm.yo_module.create")
    return fcreate(name, "dev", code, "dev_cfg") 


def create_runtime_module(target_device, code, dev_cfg, name="main"):
    fcreate = tvm._ffi.get_global_func("tvm.yo_module.create")
    return fcreate(name, target_device, code, dev_cfg)
