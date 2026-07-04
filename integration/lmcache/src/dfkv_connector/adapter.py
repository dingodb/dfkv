# SPDX-License-Identifier: Apache-2.0
"""LMCache RemoteConnector adapter for dfkv.

Wire-up in LMCache config (plugin mode, recommended):

    remote_storage_plugins: ["dfkv"]
    extra_config:
      remote_storage_plugin.dfkv.module_path: dfkv_connector.adapter
      remote_storage_plugin.dfkv.class_name:  DfkvConnectorAdapter
      remote_storage_plugin.dfkv.url:         dfkv://mds1:6700,mds2:6700/group-1

      # Optional knobs:
      remote_storage_plugin.dfkv.membership:  mds      # "mds" (default) | "static"
      remote_storage_plugin.dfkv.lib:         /path/to/libdfkv.so   # overrides $DFKV_LIB
      remote_storage_plugin.dfkv.mds_poll_ms: 3000
      # RDMA transport/tuning (v1.13.1+; go through the v2 config struct → scoped
      # env inside dfkv_open_v2, no process-wide os.environ leak; 0/None/false =
      # binary default, no env fallback on the Python side):
      remote_storage_plugin.dfkv.rdma_dev:        ib7s400p0,ib7s400p1  # comma-separated multi-rail list
      remote_storage_plugin.dfkv.rdma_depth:      4
      remote_storage_plugin.dfkv.rdma_numa:       0    # 1 = NUMA-local rail selection
      remote_storage_plugin.dfkv.require_rdma:    1    # fail if transport fell back to TCP
      remote_storage_plugin.dfkv.batch_concurrency: 8

For static membership the URL carries the member string directly, e.g.
    remote_storage_plugin.dfkv.url:        dfkv://n1=10.0.0.1:12000,n2=10.0.0.2:12000/x
    remote_storage_plugin.dfkv.membership: static

Legacy fallback: top-level ``remote_url: "dfkv://..."`` also works; per-plugin
knobs are not available on that path (defaults: membership=mds, lib from
$DFKV_LIB).

URLs MUST be ``dfkv://<endpoint>/<group>`` — no query string. Everything else
goes through extra_config above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
)
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

from .access_log import access_log
from .remote_connector import DfkvConnector

logger = init_logger(__name__)

_PLUGIN_TYPE = "dfkv"
_PLUGIN_URL_PREFIX = "plugin://"
_LEGACY_URL_PREFIX = "dfkv://"
_DEFAULT_MEMBERSHIP = "mds"
_DEFAULT_MDS_POLL_MS = 3000


def _truthy(v) -> bool:
    """Interpret an extra_config value as a boolean (mirrors the SGLang/vLLM
    connectors). Strings: ""/"0"/"false"/"no"/"off" → False; everything else
    True. Numbers/bools: plain bool()."""
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)


def _resolve_nonneg_int(extra_config: dict, prefix: str, key: str) -> int:
    """Parse extra_config['<prefix>.<key>'] as a non-negative int. Missing → 0
    (binary default). Raises ValueError on non-int or negative."""
    raw = extra_config.get(f"{prefix}.{key}")
    if raw is None:
        return 0
    try:
        val = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"extra_config['{prefix}.{key}']={raw!r} must be an integer"
        ) from e
    if val < 0:
        raise ValueError(
            f"extra_config['{prefix}.{key}']={val} must be non-negative"
        )
    return val


def _resolve_bool(extra_config: dict, prefix: str, key: str) -> bool:
    """Parse extra_config['<prefix>.<key>'] as a bool. Missing → False."""
    return _truthy(extra_config.get(f"{prefix}.{key}"))


def _extract_plugin_type(plugin_name: str) -> str:
    return plugin_name.split(".", 1)[0]


@dataclass(frozen=True)
class _PluginOverrides:
    target_url: str
    membership: str
    lib_path: Optional[str]
    mds_poll_ms: int
    rdma_depth: int = 0
    rdma_numa: int = 0
    rdma_dev: Optional[str] = None
    require_rdma: bool = False
    batch_concurrency: int = 0


class DfkvConnectorAdapter(ConnectorAdapter):
    """URL adapter for dfkv.

    Matches both ``dfkv://...`` (legacy ``remote_url`` path) and
    ``plugin://dfkv[.instance]`` (LMCache plugin path).
    """

    def __init__(self) -> None:
        with access_log("adapter.__init__", lambda: ""):
            super().__init__(_LEGACY_URL_PREFIX)

    def can_parse(self, url: str) -> bool:
        if url.startswith(_LEGACY_URL_PREFIX):
            return True
        if url.startswith(_PLUGIN_URL_PREFIX):
            pname = url[len(_PLUGIN_URL_PREFIX):]
            return _extract_plugin_type(pname) == _PLUGIN_TYPE
        return False

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        with access_log("adapter.create_connector",
                        lambda: f"url={context.url}"):
            ov = self._resolve_overrides(context)
            logger.info(
                "Creating DfkvConnector: context_url=%s target_url=%s "
                "membership=%s lib=%s mds_poll_ms=%d",
                context.url, ov.target_url, ov.membership,
                ov.lib_path or "-", ov.mds_poll_ms,
            )
            return DfkvConnector(
                url=ov.target_url,
                loop=context.loop,
                local_cpu_backend=context.local_cpu_backend,
                lib_path=ov.lib_path,
                membership=ov.membership,
                mds_poll_ms=ov.mds_poll_ms,
                rdma_depth=ov.rdma_depth,
                rdma_numa=ov.rdma_numa,
                rdma_dev=ov.rdma_dev,
                require_rdma=ov.require_rdma,
                batch_concurrency=ov.batch_concurrency,
            )

    @staticmethod
    def _resolve_overrides(context: ConnectorContext) -> _PluginOverrides:
        url = context.url

        if url.startswith(_LEGACY_URL_PREFIX):
            return _PluginOverrides(
                target_url=url, membership=_DEFAULT_MEMBERSHIP,
                lib_path=None, mds_poll_ms=_DEFAULT_MDS_POLL_MS,
            )

        plugin_name = getattr(context, "plugin_name", None) or url[
            len(_PLUGIN_URL_PREFIX):
        ]
        prefix = f"remote_storage_plugin.{plugin_name}"

        extra_config = (
            getattr(context.config, "extra_config", None)
            if context.config is not None else None
        ) or {}

        target_url = extra_config.get(f"{prefix}.url")
        if not target_url:
            raise ValueError(
                f"Plugin mode requires extra_config['{prefix}.url'] to be a "
                "dfkv:// URL (e.g. 'dfkv://mds1:6700/group-1')"
            )
        if not isinstance(target_url, str) or not target_url.startswith(
            _LEGACY_URL_PREFIX
        ):
            raise ValueError(
                f"extra_config['{prefix}.url']={target_url!r} must be a string "
                f"starting with {_LEGACY_URL_PREFIX!r}"
            )

        membership = extra_config.get(f"{prefix}.membership", _DEFAULT_MEMBERSHIP)
        if membership not in ("mds", "static"):
            raise ValueError(
                f"extra_config['{prefix}.membership']={membership!r} must be "
                "'mds' or 'static'"
            )

        lib_path = extra_config.get(f"{prefix}.lib")
        if lib_path is not None and not isinstance(lib_path, str):
            raise ValueError(
                f"extra_config['{prefix}.lib']={lib_path!r} must be a "
                "filesystem path string"
            )

        raw_poll = extra_config.get(f"{prefix}.mds_poll_ms")
        mds_poll_ms = _DEFAULT_MDS_POLL_MS
        if raw_poll is not None:
            try:
                mds_poll_ms = int(raw_poll)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"extra_config['{prefix}.mds_poll_ms']={raw_poll!r} must be "
                    "an integer"
                ) from e
            if mds_poll_ms <= 0:
                raise ValueError(
                    f"extra_config['{prefix}.mds_poll_ms']={mds_poll_ms} must be "
                    "positive"
                )

        # RDMA transport/tuning knobs (v2 config struct → scoped env inside
        # dfkv_open_v2; no process-wide os.environ leak). 0/NULL/False = binary
        # default, no env fallback on the Python side — extra_config is the
        # single source. rdma_dev is a comma-separated multi-rail list.
        rdma_depth = _resolve_nonneg_int(extra_config, prefix, "rdma_depth")
        rdma_numa = _resolve_bool(extra_config, prefix, "rdma_numa")
        rdma_dev = extra_config.get(f"{prefix}.rdma_dev")
        if rdma_dev is not None and not isinstance(rdma_dev, str):
            raise ValueError(
                f"extra_config['{prefix}.rdma_dev']={rdma_dev!r} must be a "
                "comma-separated device list string"
            )
        require_rdma = _resolve_bool(extra_config, prefix, "require_rdma")
        batch_concurrency = _resolve_nonneg_int(
            extra_config, prefix, "batch_concurrency")

        return _PluginOverrides(
            target_url=target_url,
            membership=membership,
            lib_path=lib_path or None,
            mds_poll_ms=mds_poll_ms,
            rdma_depth=rdma_depth,
            rdma_numa=1 if rdma_numa else 0,
            rdma_dev=rdma_dev or None,
            require_rdma=require_rdma,
            batch_concurrency=batch_concurrency,
        )
