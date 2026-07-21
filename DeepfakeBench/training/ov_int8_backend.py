"""Minimal server-side OpenVINO INT8 backend for app3.py.

Adds a ``model_type=custom_int8`` path on ``/check_frame_batch`` so the cloud
server can return the SAME per-frame probabilities as the local NPU build's INT8
model. That lets us tune the frame threshold once against numbers that match the
field client (remote<->local parity — see WMA_ACCURACY_PLAN_A).

Intentionally self-contained (NO license / TPM / blob-cache): the cloud model is
already server-side, so no client-distribution protection is needed — we load the
plain OpenVINO IR directly. This is the ONLY thing app3.py imports for INT8, and
it is imported lazily (only when INT8_MODEL_PATH is set), so app3.py still imports
and serves FP32 on hosts where ``openvino`` is not installed.

Interface contract (identical to the PyTorch detector app3 already calls):

    model({'image': <uint8 NHWC BGR ndarray [N, 224, 224, 3]>}, inference=True)
        -> {'prob': torch.Tensor([N])}

The INT8 IR is quantized WITH preprocessing baked into the graph, so it consumes
uint8 NHWC BGR and internally does BGR->RGB + uint8->float/255 + CLIP normalize.
The caller must NOT colour-swap or normalize — app3.py feeds the resized uint8
BGR crop straight through on the ``baked_int8`` path.
"""
from __future__ import annotations

import numpy as np
import torch
import openvino as ov


class OVInt8Model:
    """Drop-in replacement for the PyTorch video model, backed by an OpenVINO IR.

    Exposes the same ``__call__({'image': ...}, inference=True) -> {'prob': ...}``
    interface as the FP32 detector. The ``expects_uint8_bgr_nhwc = True`` marker
    tells the app3 endpoint to feed uint8 NHWC BGR frames (the baked-preprocessing
    input contract) instead of the CLIP-normalized float NCHW tensor the FP32 path
    builds.
    """

    expects_uint8_bgr_nhwc = True

    def __init__(self, compiled_model: "ov.CompiledModel") -> None:
        self._model = compiled_model
        self._output = compiled_model.output(0)

    def __call__(self, data_dict: dict, inference: bool = True, batch_inference: bool = True) -> dict:
        # Score ONE frame per infer call (batch=1) to match the local build's
        # OPENVINO_BATCH_SIZE=1 wrapper EXACTLY. OpenVINO's CPU plugin returns
        # batch-size-dependent probabilities at large batches (measured: a
        # 170-frame batch shifted some probs by up to ~0.08 vs batch=1; batches
        # <=64 were bit-identical). Per-frame scoring keeps the cloud probs
        # identical to the local per-frame inference regardless of how many frames
        # WMA sends per request, and is immune to any machine-specific tiling
        # threshold. Latency is irrelevant on this parity endpoint.
        image = data_dict["image"]          # uint8 NHWC BGR [N, 224, 224, 3]
        n = int(image.shape[0])
        probs = np.empty(n, dtype=np.float32)
        for i in range(n):
            out = self._model(image[i:i + 1])[self._output]
            probs[i] = float(np.asarray(out).reshape(-1)[0])
        # Return a torch tensor so the caller's .detach().squeeze().cpu().numpy()
        # chain works exactly as it does for the PyTorch model output.
        return {"prob": torch.from_numpy(probs)}


def load_int8_model(model_xml_path: str, device: str = "CPU") -> OVInt8Model:
    """Compile the plain INT8 OpenVINO IR on ``device`` (CPU on the cloud host).

    The matching ``.bin`` must sit next to the ``.xml``. No encryption/TPM — this
    is the server-side plain IR. On a GCP host the only OpenVINO-usable device is
    CPU (the Tesla T4 is CUDA, which OpenVINO cannot target); values are
    deterministic and match the local INT8-on-CPU reference bit-for-bit.
    """
    core = ov.Core()
    model = core.read_model(model_xml_path)
    compiled = core.compile_model(model, device)
    return OVInt8Model(compiled)
