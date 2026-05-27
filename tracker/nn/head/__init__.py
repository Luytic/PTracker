from __future__ import annotations

from tracker.nn.head.ban_v2 import DepthwiseBAN as DepthwiseBANv2
from tracker.nn.head.ban_v2 import UPChannelBAN as UPChannelBANv2
from tracker.nn.head.ban_v3 import DepthwiseBAN as DepthwiseBANv3
from tracker.nn.head.ban_v3 import UPChannelBAN as UPChannelBANv3

_VARIANTS = {
    "v2": {"DepthwiseBAN": DepthwiseBANv2, "UPChannelBAN": UPChannelBANv2},
    "v3": {"DepthwiseBAN": DepthwiseBANv3, "UPChannelBAN": UPChannelBANv3},
}


def get_ban_head(name: str, *, variant: str = "v2", **kwargs):
    try:
        return _VARIANTS[variant][name](**kwargs)
    except KeyError as exc:
        raise KeyError(f"Unknown BAN head {name!r} for variant {variant!r}") from exc
