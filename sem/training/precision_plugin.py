import torch
import torch.nn as nn
from typing import Optional, Any
from pytorch_lightning.plugins.precision import MixedPrecision


class SEMPrecisionPlugin(MixedPrecision):
    def __init__(self, precision: str, device: str, scaler: Optional[Any] = None):
        super().__init__(precision=precision, device=device, scaler=scaler)
        self._propagator_modules = []
        self._device_type = device

    def setup_module(self, module: nn.Module) -> None:
        self._propagator_modules = []
        for name, submodule in module.named_modules():
            if submodule.__class__.__name__ == "CayleySolitonPropagator":
                self._propagator_modules.append(submodule)

    def forward_context(self) -> Any:
        base_ctx = super().forward_context()
        return _PropagatorExclusionContext(
            base_ctx,
            self._propagator_modules,
            self._device_type,
        )


class _PropagatorExclusionContext:
    def __init__(self, base_ctx: Any, propagator_modules: list, device_type: str):
        self.base_ctx = base_ctx
        self.propagator_modules = propagator_modules
        self.device_type = device_type
        self._original_forwards = {}

    def __enter__(self):
        self.base_ctx.__enter__()
        for propagator in self.propagator_modules:
            original_forward = propagator.forward

            def make_wrapped_forward(orig_fwd):
                def wrapped_forward(*args, **kwargs):
                    with torch.autocast(device_type=self.device_type, enabled=False):
                        return orig_fwd(*args, **kwargs)

                return wrapped_forward

            self._original_forwards[id(propagator)] = original_forward
            propagator.forward = make_wrapped_forward(original_forward)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for propagator in self.propagator_modules:
            if id(propagator) in self._original_forwards:
                propagator.forward = self._original_forwards[id(propagator)]

        self._original_forwards.clear()
        return self.base_ctx.__exit__(exc_type, exc_val, exc_tb)
