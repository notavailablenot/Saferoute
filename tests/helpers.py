"""Tiny stand-in ONNX models with the same I/O contracts as the real SafeRoute models,
so the engines and the API can be tested without the trained weights or a GPU."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def tiny_classifier(path: Path, n_classes: int = 11, favour: int | None = None):
    """[1,3,64,64] -> [1,n_classes] logits. With `favour`, that class always wins strongly."""
    w = np.random.rand(3, n_classes).astype(np.float32) * 0.01
    b = np.zeros(n_classes, dtype=np.float32)
    if favour is not None:
        b[favour] = 20.0
    g = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["input"], ["gap"]),
         helper.make_node("Flatten", ["gap"], ["flat"]),
         helper.make_node("MatMul", ["flat", "W"], ["mm"]),
         helper.make_node("Add", ["mm", "B"], ["logits"])],
        "tiny_cls", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, n_classes])],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)


def tiny_detector(path: Path, boxes_640: list[tuple[float, float, float, float, float]],
                  n_anchors: int = 8400):
    """[1,3,640,640] -> [1,5,n_anchors] like a 1-class YOLO11 export.

    `boxes_640` holds (cx, cy, w, h, score) in letterboxed 640-pixel coordinates; every other
    anchor has score 0. The image is ignored (output = constant + 0 * mean(image)) so the
    output is fully predictable for tests.
    """
    out = np.zeros((1, 5, n_anchors), dtype=np.float32)
    for i, (cx, cy, w, h, s) in enumerate(boxes_640):
        out[0, :, i] = [cx, cy, w, h, s]
    g = helper.make_graph(
        [helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
         helper.make_node("Mul", ["m", "zero"], ["z"]),
         helper.make_node("Add", ["C", "z"], ["output0"])],
        "tiny_det", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, n_anchors])],
        [numpy_helper.from_array(out, "C"), numpy_helper.from_array(np.array(0, np.float32), "zero")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, path)
