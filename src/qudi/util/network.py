# -*- coding: utf-8 -*-
"""
RPyC helper methods for qudi.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-core/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

__all__ = ['netobtain', 'connect_remote_module']

from rpyc import ssl_connect as _ssl_connect
from rpyc import Connection as _Connection
from rpyc.core.netref import BaseNetref as _BaseNetref
from rpyc.utils.classic import obtain as _obtain
from urllib.parse import urlparse as _urlparse
from typing import Optional, Tuple, Dict, Any


def netobtain(obj):
    """
    """
    if isinstance(obj, _BaseNetref):
        return _obtain(obj)
    return obj


def connect_remote_module(remote_url: str,
                          certfile: Optional[str] = None,
                          keyfile: Optional[str] = None,
                          protocol_config: Optional[Dict[str, Any]] = None
                          ) -> Tuple[_Connection, str]:
    """ Connects via RPyC to a RemoteModulesServer instance and returns the rpyc.Connection object
    along with the remote module name given in the URL (if present).
    """
    if protocol_config is None:
        protocol_config = {'allow_all_attrs'     : True,
                           'allow_setattr'       : True,
                           'allow_delattr'       : True,
                           'allow_pickle'        : True,
                           'sync_request_timeout': 3600}
    parsed = _urlparse(remote_url)
    connection = _ssl_connect(host=parsed.hostname,
                              port=parsed.port,
                              config=protocol_config,
                              certfile=certfile,
                              keyfile=keyfile)
    return connection, parsed.path.replace('/', '')
