# Owner(s): ["module: dynamo"]

import copy
import os
import sys
import tempfile

import torch
import torch.distributed as dist
from torch._guards import tracing, TracingContext
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.regional_inductor import (
    _functionalize_inplace_collectives,
    regional_inductor,
)
from torch.testing._internal.common_utils import run_tests, TestCase


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


def _f(t):
    t = t.clone()
    dist.all_reduce(t)
    return t + 1


def _make_fx_with_allreduce():
    return make_fx(_f)(torch.ones(4))


class TestRegionalInductorCollectives(TestCase):
    def setUp(self):
        super().setUp()
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29516")
        # Single-rank Gloo lets us exercise real all_reduce numerics
        # in-process without spawning a second worker. With world_size=1,
        # ``all_reduce(t, op=SUM)`` is the identity, so any deviation in the
        # rewrite is observable as a numeric mismatch against eager.
        self._store_path = tempfile.mktemp()
        dist.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            store=dist.FileStore(self._store_path, 1),
        )

    def tearDown(self):
        try:
            dist.destroy_process_group()
        except AssertionError:
            pass
        super().tearDown()

    def test_functionalize_inplace_allreduce(self):
        gm = _make_fx_with_allreduce()
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    _torchbind_obj0 = self._torchbind_obj0
    _torchbind_obj1 = self._torchbind_obj1
    allreduce_ = torch.ops.c10d.allreduce_.default([clone], _torchbind_obj0, _torchbind_obj1, None, False);  clone = _torchbind_obj0 = _torchbind_obj1 = None
    getitem = allreduce_[0]
    getitem_1 = getitem[0];  getitem = None
    getitem_2 = allreduce_[1];  allreduce_ = getitem_2 = None
    add = torch.ops.aten.add.Tensor(getitem_1, 1);  getitem_1 = None
    return add""",
        )
        self.assertTrue(any(k.startswith("_torchbind_obj") for k in gm.__dict__))

        _functionalize_inplace_collectives(gm)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, t_1):
    clone = torch.ops.aten.clone.default(t_1);  t_1 = None
    all_reduce_default = torch.ops._c10d_functional.all_reduce.default(clone, 'sum', '0')
    wait_tensor_default = torch.ops._c10d_functional.wait_tensor.default(all_reduce_default);  all_reduce_default = None
    copy__default = torch.ops.aten.copy_.default(clone, wait_tensor_default);  clone = copy__default = None
    add = torch.ops.aten.add.Tensor(wait_tensor_default, 1);  wait_tensor_default = None
    return add""",
        )
        # Torchbind ProcessGroup / ReduceOp attrs are stripped so downstream
        # consumers can deepcopy the GraphModule.
        self.assertFalse(any(k.startswith("_torchbind_obj") for k in gm.__dict__))
        copy.deepcopy(gm)

        # Numerics: rewritten graph must match eager.
        x = torch.arange(4, dtype=torch.float32)
        self.assertEqual(gm(x), _f(x))

    def test_regional_inductor_with_dist_all_reduce(self):
        gm = _make_fx_with_allreduce()
        for node in gm.graph.nodes:
            if node.op not in ("placeholder", "output"):
                node.meta.setdefault("custom", {})["compile_with_inductor"] = {
                    "inductor_configs": {}
                }

        fake_mode = next(
            n.meta["val"].fake_mode
            for n in gm.graph.nodes
            if n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Tensor)
        )

        with tracing(TracingContext(fake_mode)):
            compiled_gm = regional_inductor(gm)

        self.assertNotIn("c10d.allreduce_", compiled_gm.code)
        x = torch.arange(4, dtype=torch.float32)
        # ``regional_inductor`` switches the submodule to boxed calling
        # convention; pass inputs as a list (the callee clears it).
        self.assertEqual(compiled_gm([x]), _f(x))


if __name__ == "__main__":
    run_tests()
