import tvm

class YoTarget:
    cur, last = None, None

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.last = YoTarget.cur
        YoTarget.cur = self
        return self

    def __exit__(self, ptype, value, trace):
        YoTarget.cur = self.last

    @classmethod
    def get_target(cls, tgt_name):
        return cls(tgt_name)

    def get_ops(self):
        return []

def GetYoTarget(model="yo-dev", options=None):
    opts = ["-keys=yo,numpy,cpu", "-model=%s" % model]
    opts = []
    opts = tvm.target.target._merge_opts(opts, options)
    target = tvm.target.Target(" ".join(["yox"] + opts))
    tvm.target.Target.list_kinds()
    return target