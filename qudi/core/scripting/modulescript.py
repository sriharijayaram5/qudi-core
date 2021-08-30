# -*- coding: utf-8 -*-

"""
This file contains a basic script class to run with qudi module dependencies as well as various
helper classes to run and manage these scripts.

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

__all__ = ['import_module_script', 'ModuleScript', 'ModuleScriptsTableModel',
           'ModuleScriptInterrupted']

import importlib
import copy
import inspect
from abc import abstractmethod
from uuid import uuid4
from PySide2 import QtCore
from logging import getLogger, Logger
from typing import Mapping, Any, Type, Optional

from qudi.core.meta import QudiObjectMeta
from qudi.util.models import DictTableModel
from qudi.util.mutex import Mutex


class ModuleScriptInterrupted(Exception):
    """ Custom exception class to indicate that a ModuleScript execution has been interrupted.
    """
    pass


class ModuleScript(QtCore.QObject, metaclass=QudiObjectMeta):
    """
    The only part that can be interrupted is the _run() method.
    The implementations must occasionally call _check_interrupt() to raise an exception at that
    point if an interrupt is requested.
    """
    # Declare all module connectors used in this script here

    sigFinished = QtCore.Signal(object, str, bool)  # result, ID, success

    # FIXME: This __new__ implementation has the sole purpose to circumvent a known PySide2(6) bug.
    #  See https://bugreports.qt.io/browse/PYSIDE-1434 for more details.
    def __new__(cls, *args, **kwargs):
        abstract = getattr(cls, '__abstractmethods__', frozenset())
        if abstract:
            raise TypeError(f'Can\'t instantiate abstract class "{cls.__name__}" '
                            f'with abstract methods {set(abstract)}')
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent=parent)

        # Create a copy of the _meta class dict and attach it to this instance
        self._meta = copy.deepcopy(self._meta)
        # Create unique ID string and attach to _meta dict
        self._meta['uuid'] = uuid4()
        # set instance attributes according to connector meta objects
        for attr_name, conn in self._meta['connectors'].items():
            setattr(self, attr_name, conn)

        self._thread_lock = Mutex()

        # script arguments and result cache
        self.args = tuple()
        self.kwargs = dict()
        self.result = None

        # Status flags
        self._stop_requested = False
        self._success = False
        self._running = False

    @property
    def interrupted(self):
        with self._thread_lock:
            return self._stop_requested

    def interrupt(self):
        with self._thread_lock:
            self._stop_requested = True

    def _check_interrupt(self) -> None:
        """ Implementations of _run should occasionally call this method in order to break
        execution early if another thread has interrupted this script in the meantime.
        """
        if self.interrupted:
            raise ModuleScriptInterrupted

    @property
    def id(self) -> str:
        """ Read-only unique id (uuid4) of this script instance.

        @return str: ID of this script instance
        """
        return str(self._meta['uuid'])

    @property
    def log(self) -> Logger:
        """ Returns a logger object.
        DO NOT OVERRIDE IN SUBCLASS!

        @return Logger: Logger object for this script class
        """
        return getLogger(f'{self.__module__}.{self.__class__.__name__}')

    @property
    def running(self) -> bool:
        with self._thread_lock:
            return self._running

    @property
    def success(self) -> bool:
        with self._thread_lock:
            return self._success

    @property
    def call_signature(self) -> inspect.Signature:
        """ Signature of the _run method implementation.
        Override in subclass if you want anything else than this default implementation.
        Make sure custom implementations of this property are compatible with _run.
        """
        return inspect.signature(self._run)

    def __call__(self, *args, **kwargs) -> Any:
        """ Convenience magic method to run this script like a function
        DO NOT OVERRIDE IN SUBCLASS!

        Arguments are passed directly to _run() method.

        @return object: Result of the script method
        """
        self.args = args
        self.kwargs = kwargs
        self.run()
        return self.result

    @QtCore.Slot()
    def run(self) -> None:
        """ Check run prerequisites and execute _run method with pre-cached arguments.
        DO NOT OVERRIDE IN SUBCLASS!
        """
        self.result = None
        with self._thread_lock:
            self._success = False
            self._running = True
        self.log.debug(f'Starting to run ModuleScript "{self.__class__.__name__}" with positional '
                       f'arguments {self.args} and keyword arguments {self.kwargs}.')
        # Emit finished signal even if script execution fails. Check success flag.
        try:
            self.result = self._run(*self.args, **self.kwargs)
            with self._thread_lock:
                self._success = True
        except ModuleScriptInterrupted:
            self.log.debug(f'Interrupted main method of ModuleScript "{self.__class__.__name__}".')
        finally:
            with self._thread_lock:
                self._running = False
                self.sigFinished.emit(self.result, self.id, self._success)

    def connect_modules(self, connector_targets: Mapping[str, Any]) -> None:
        """ Connects given modules (values) to their respective Connector (keys).

        DO NOT CALL THIS METHOD UNLESS YOU KNOW WHAT YOU ARE DOING!
        """
        # Sanity checks
        conn_names = set(conn.name for conn in self._meta['connectors'].values())
        mandatory_conn = set(
            conn.name for conn in self._meta['connectors'].values() if not conn.optional
        )
        configured_conn = set(connector_targets)
        if not configured_conn.issubset(conn_names):
            raise KeyError(f'Mismatch of connectors in configuration {configured_conn} and '
                           f'Connector meta objects {conn_names}.')
        if not mandatory_conn.issubset(configured_conn):
            raise ValueError(f'Not all mandatory connectors are specified.\n'
                             f'Mandatory connectors are: {mandatory_conn}')

        # Iterate through module connectors and connect them if possible
        for conn in self._meta['connectors'].values():
            target = connector_targets.get(conn.name, None)
            if target is None:
                continue
            if conn.is_connected:
                raise RuntimeError(f'Connector "{conn.name}" already connected.\n'
                                   f'Call "disconnect_modules()" before trying to reconnect.')
            conn.connect(target)

    def disconnect_modules(self) -> None:
        """ Disconnects all Connector instances for this object.

        DO NOT CALL THIS METHOD UNLESS YOU KNOW WHAT YOU ARE DOING!
        """
        for conn in self._meta['connectors'].values():
            conn.disconnect()

    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        """ The actual script to be run. Implement only this method in a subclass.
        """
        raise NotImplementedError(f'No _run() method implemented for "{self.__class__.__name__}".')


def import_module_script(module: str, cls: str,
                         reload: Optional[bool] = True) -> Type[ModuleScript]:
    """ Helper function to import ModuleScript sub-classes by name from a given module.
    Reloads the module to import from by default.
    """
    mod = importlib.import_module(module)
    if reload:
        importlib.reload(mod)
    script = getattr(mod, cls)
    if not issubclass(script, ModuleScript):
        raise TypeError(f'Module script to import must be a subclass of {__name__}.ModuleScript')
    return script


class ModuleScriptsTableModel(DictTableModel):
    """ Qt compatible table model holding all configured and available ModuleScript subclasses.
    """
    def __init__(self, script_config: Optional[Mapping[str, dict]] = None):
        super().__init__(headers='Module Scripts')
        if script_config is None:
            script_config = dict()
        for name, config in script_config.items():
            self.register_script(name, config)

    def register_script(self, name: str, config: dict) -> None:
        if name in self:
            raise KeyError(f'Multiple module script with name "{name}" configured.')
        module, cls = config['module.Class'].rsplit('.', 1)
        self[name] = import_module_script(module, cls, reload=True)