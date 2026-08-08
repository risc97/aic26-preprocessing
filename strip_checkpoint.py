import torch, numpy as np
import numpy._core.multiarray as m

torch.serialization.add_safe_globals([
    (m.scalar, "numpy._core.multiarray.scalar"),
    np.dtype, np.dtypes.Float64DType, np.dtypes.Int64DType,
])
ck = torch.load("data/checkpoints/c2lip_npc_xac.pt", map_location="cpu", weights_only=True)
sd = {k.removeprefix("module."): v for k, v in ck["state_dict"].items()}
torch.save(sd, "data/checkpoints/c2lip.pt")