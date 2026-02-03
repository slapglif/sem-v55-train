import torch
from pytorch_lightning.accelerators import Accelerator
from lightning_fabric.utilities.exceptions import MisconfigurationException


class XPUAccelerator(Accelerator):
    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise MisconfigurationException(
                f"Device should be XPU, got {device} instead."
            )
        if device.index is None:
            torch.xpu.set_device(0)
        else:
            torch.xpu.set_device(device)

    def get_device_stats(self, device: torch.device) -> dict:
        if hasattr(torch.xpu, "memory_stats"):
            return torch.xpu.memory_stats(device)
        return {}

    def teardown(self) -> None:
        if hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()

    @staticmethod
    def parse_devices(devices):
        if devices is None or devices == "auto":
            return list(range(XPUAccelerator.auto_device_count()))
        if isinstance(devices, int):
            return list(range(devices))
        return devices

    @staticmethod
    def get_parallel_devices(devices) -> list[torch.device]:
        return [torch.device("xpu", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count() if hasattr(torch, "xpu") else 0

    @staticmethod
    def is_available() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    def name() -> str:
        return "xpu"
