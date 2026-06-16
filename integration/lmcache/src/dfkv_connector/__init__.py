# SPDX-License-Identifier: Apache-2.0
"""dfkv connector for LMCache.

Exports the URL adapter (register it as a remote_storage_plugin) and the
RemoteConnector implementation. dfkv is reached via the C ABI (libdfkv.so,
loaded with ctypes) — there is no native CPython extension to build.
"""

from .adapter import DfkvConnectorAdapter
from .remote_connector import DfkvConnector

__all__ = ["DfkvConnector", "DfkvConnectorAdapter"]
