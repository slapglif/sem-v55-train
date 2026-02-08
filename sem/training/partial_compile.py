import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SelectiveCompileWrapper(nn.Module):
    """Wrap specific submodules with torch.compile to optimize float32 paths
    while leaving complex64 paths eager (avoiding stride errors).
    """

    def __init__(
        self, model: nn.Module, compile_list: list[str], mode: str = "reduce-overhead"
    ):
        super().__init__()
        self.model = model

        for name in compile_list:
            try:
                # Find submodule
                if "." in name:
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent_name = ""
                    attr_name = name
                    parent = model

                submodule = getattr(parent, attr_name)

                # Compile it
                compiled = torch.compile(submodule, mode=mode, fullgraph=False)

                # Replace
                setattr(parent, attr_name, compiled)
                logger.info(f"✓ Partially compiled: {name}")

            except Exception as e:
                logger.warning(f"✗ Failed to compile {name}: {e}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
