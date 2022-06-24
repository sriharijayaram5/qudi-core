# -*- coding: utf-8 -*-
"""
This file contains the Qudi Manager class.

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

import os
import importlib
import copy
import weakref
import fysom

from typing import FrozenSet, Iterable, Optional, Any, Union, Mapping, Dict, List, Set
from functools import partial
from PySide2 import QtCore

from qudi.util.mutex import Mutex   # provides access serialization between threads
from qudi.core.logger import get_logger
from qudi.core.servers import get_remote_module_instance
from qudi.core.module import Base
from qudi.util.paths import get_module_app_data_path

logger = get_logger(__name__)


class ManagedModule(QtCore.QObject):
    """ Object representing a qudi module (gui, logic or hardware) to be managed by the qudi Manager
     object. Contains status properties and handles initialization, state transitions and
     connection of the module.
    """
    sigStateChanged = QtCore.Signal(object)  # self
    sigAppDataChanged = QtCore.Signal(object)  # self

    __state_poll_interval = 1  # Max interval in seconds to poll module_state of remote modules

    def __init__(self,
                 name: str,
                 base: str,
                 configuration: Mapping[str, Any],
                 qudi_main_ref: weakref.ref,
                 parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent=parent)
        if not name or not isinstance(name, str):
            raise ValueError('Module name must be a non-empty string.')
        if base not in ['gui', 'logic', 'hardware']:
            raise ValueError('Module base must be one of ["gui", "logic", "hardware"].')
        if self.thread() is not QtCore.QCoreApplication.instance().thread():
            raise RuntimeError('ManagedModules can only be owned by the application main thread.')

        self._lock = Mutex()

        # Weak reference to qudi main instance
        if not isinstance(qudi_main_ref, weakref.ref):
            qudi_main_ref = weakref.ref(qudi_main_ref)
        self._qudi_main_ref = qudi_main_ref
        self._name = name  # Each qudi module needs a unique string identifier
        self._base = base  # Remember qudi module base
        self._instance = None  # Holds the qudi module instance after loading the module


        # Sort out configuration dict
        # Extract module and class name
        self._module, self._class = configuration.get('module.Class', '.').rsplit('.', 1)
        # Remember connections by name
        self._connections = configuration.get('connect', dict())
        # See if remote access to this module is allowed (NOT allowed by default)
        self._allow_remote = configuration.get('allow_remote', False)
        # Get module options
        self._options = configuration.get('options', dict())
        # Extract remote modules URL and certificate if this module is run on a remote machine
        self._remote_url = configuration.get('remote_url', '')
        self._certfile = configuration.get('certfile', '')
        self._keyfile = configuration.get('keyfile', '')
        # Do not propagate remote access
        if self._remote_url:
            self._allow_remote = False

        self._required_modules = frozenset()
        self._dependent_modules = frozenset()

        self.__poll_timer = None
        self.__last_state = None

        if self.is_remote:
            self._status_file_path = ''
        else:
            self._status_file_path = get_module_app_data_path(self._class, self._base, self._name)

    def __call__(self):
        return self.instance

    @property
    def name(self) -> str:
        return self._name

    @property
    def base(self) -> str:
        return self._base

    @property
    def class_name(self) -> str:
        return self._class

    @property
    def module_name(self) -> str:
        return self._module

    @property
    def options(self) -> Dict[str, Any]:
        return self._options

    @property
    def instance(self) -> Base:
        return self._instance

    @property
    def status_file_path(self) -> str:
        return self._status_file_path

    @property
    def is_loaded(self) -> bool:
        return self.state != 'not loaded'

    @property
    def is_active(self) -> bool:
        return self.state not in ['not loaded', 'deactivated']

    @property
    def is_busy(self) -> bool:
        return self.state not in ['not loaded', 'deactivated', 'idle']

    @property
    def is_remote(self) -> bool:
        return bool(self._remote_url)

    @property
    def allow_remote(self) -> bool:
        return self._allow_remote

    @property
    def remote_url(self) -> str:
        return self._remote_url

    @property
    def keyfile(self) -> str:
        return self._keyfile

    @property
    def certfile(self) -> str:
        return self._certfile

    @property
    def state(self) -> str:
        try:
            return self._instance.module_state()
        except AttributeError:
            return 'not loaded'

    @property
    def connections(self):
        return self._connections.copy()

    @property
    def required_modules(self) -> FrozenSet[weakref.ref]:
        return self._required_modules

    @required_modules.setter
    def required_modules(self, module_references: Iterable[weakref.ref]) -> None:
        module_references = frozenset(module_references)
        if not all(isinstance(mod(), ManagedModule) for mod in module_references):
            raise TypeError('"required_modules" references must point to ManagedModule instances')
        self._required_modules = module_references

    @property
    def dependent_modules(self) -> FrozenSet[weakref.ref]:
        return self._dependent_modules

    @dependent_modules.setter
    def dependent_modules(self, module_references: Iterable[weakref.ref]) -> None:
        module_references = frozenset(module_references)
        if not all(isinstance(mod(), ManagedModule) for mod in module_references):
            raise TypeError('"dependent_modules" references must point to ManagedModule instances')
        self._dependent_modules = module_references

    @property
    def ranking_active_dependent_modules(self) -> Set[weakref.ref]:
        active_dependent_modules = set()
        for module_ref in self.dependent_modules:
            module = module_ref()
            if (module is not None) and module.is_active:
                active_modules = module.ranking_active_dependent_modules
                if active_modules:
                    active_dependent_modules.update(active_modules)
                else:
                    active_dependent_modules.add(module_ref)
        return active_dependent_modules

    @property
    def thread_name(self) -> str:
        return f'mod-{self.base}-{self.name}'

    @property
    def has_app_data(self) -> bool:
        return os.path.exists(self.status_file_path)

    @QtCore.Slot()
    def clear_app_data(self) -> None:
        try:
            os.remove(self.status_file_path)
        except OSError:
            pass
        finally:
            self.sigAppDataChanged.emit(self.base, self.name, self.has_app_data)

    @QtCore.Slot()
    def activate(self) -> None:
        module_loc = self.remote_url if self.is_remote else f'{self.module_name}.{self.class_name}'
        # Switch to the main thread if this method was called from another thread
        if QtCore.QThread.currentThread() is not self.thread():
            QtCore.QMetaObject.invokeMethod(self, 'activate', QtCore.Qt.BlockingQueuedConnection)
            if not self.is_active:
                raise RuntimeError(
                    f'Failed to activate {self.base} module "{self.name}" from {module_loc}'
                )
            return

        with self._lock:
            self._activate()

    def _activate(self) -> None:
        # Return early if already active
        if self.is_active:
            # If it is a GUI module, show it again.
            if self.base == 'gui':
                self._instance.show()
            return

        if not self.is_loaded:
            self._load()

        # Recursive activation of required modules
        for module_ref in self.required_modules:
            module = module_ref()
            if module is None:
                raise RuntimeError(
                    f'Dead required module weakref encountered in ManagedModule "{self.name}"'
                )
            module.activate()

        logger.info(f'Activating {self.base} module "{self.name}" from {module_loc}')

        # Establish module interconnections via Connector meta object in qudi module instance
        self._connect()

        # Activate this module
        if self._instance.module_threaded:
            thread_name = self.thread_name
            thread_manager = self._qudi_main_ref().thread_manager
            thread = thread_manager.get_new_thread(thread_name)
            self._instance.moveToThread(thread)
            thread.start()
            try:
                QtCore.QMetaObject.invokeMethod(self._instance.module_state,
                                                'activate',
                                                QtCore.Qt.BlockingQueuedConnection)
            finally:
                # Cleanup if activation was not successful
                if not self.is_active:
                    QtCore.QMetaObject.invokeMethod(self._instance,
                                                    'move_to_main_thread',
                                                    QtCore.Qt.BlockingQueuedConnection)
                    thread_manager.quit_thread(thread_name)
                    thread_manager.join_thread(thread_name)
        else:
            try:
                self._instance.module_state.activate()
            except fysom.Canceled:
                pass

        self.__last_state = self.state
        self.sigStateChanged.emit(self.base, self.name, self.__last_state)
        self.sigAppDataChanged.emit(self.base, self.name, self.has_app_data)

        # Raise exception if by some reason no exception propagated to here and the activation
        # is still unsuccessful.
        if not self.is_active:
            try:
                self._disconnect()
            except:
                pass
            raise RuntimeError(
                f'Failed to activate {self.base} module "{self.name}" from {module_loc}'
            )

        if self.is_remote:
            self.__poll_timer = QtCore.QTimer(self)
            self.__poll_timer.setInterval(int(round(self.__state_poll_interval * 1000)))
            self.__poll_timer.setSingleShot(True)
            self.__poll_timer.timeout.connect(self._poll_module_state)
            self.__poll_timer.start()
        else:
            self._instance.module_state.sigStateChanged.connect(self._state_change_callback)

    @QtCore.Slot()
    def _poll_module_state(self):
        current_state = self.state
        if current_state != self.__last_state:
            self.__last_state = current_state
            self.sigStateChanged.emit(self.base, self.name, current_state)
        try:
            self.__poll_timer.start()
        except AttributeError:
            pass

    @QtCore.Slot()
    def _state_change_callback(self, event=None):
        self.sigStateChanged.emit(self.base, self.name, self.state)

    @QtCore.Slot()
    def deactivate(self) -> None:
        module_loc = self.remote_url if self.is_remote else f'{self.module_name}.{self.class_name}'
        # Switch to the main thread if this method was called from another thread
        if QtCore.QThread.currentThread() is not self.thread():
            QtCore.QMetaObject.invokeMethod(self, 'deactivate', QtCore.Qt.BlockingQueuedConnection)
            if self.is_active:
                raise RuntimeError(
                    f'Failed to deactivate {self.base} module "{self.name}" from {module_loc}'
                )
            return

        with self._lock:
            self._deactivate()

    def _deactivate(self) -> None:
        if not self.is_active:
            return

        # Recursively deactivate dependent modules
        for module_ref in self.dependent_modules:
            module = module_ref()
            if module is None:
                raise RuntimeError(
                    f'Dead dependent module weakref encountered in ManagedModule "{self.name}".'
                )
            module.deactivate()

        logger.info(f'Deactivating {self.base} module "{self.name}" from {module_loc}')

        # Disable state updated
        try:
            self.__poll_timer.stop()
            self.__poll_timer.timeout.disconnect()
        except AttributeError:
            self._instance.module_state.sigStateChanged.disconnect(self._state_change_callback)
        finally:
            self.__poll_timer = None

        # Actual deactivation of this module
        if self._instance.is_module_threaded:
            thread_name = self.thread_name
            thread_manager = self._qudi_main_ref().thread_manager
            try:
                QtCore.QMetaObject.invokeMethod(self._instance.module_state,
                                                'deactivate',
                                                QtCore.Qt.BlockingQueuedConnection)
            finally:
                QtCore.QMetaObject.invokeMethod(self._instance,
                                                'move_to_main_thread',
                                                QtCore.Qt.BlockingQueuedConnection)
                thread_manager.quit_thread(thread_name)
                thread_manager.join_thread(thread_name)
        else:
            try:
                self._instance.module_state.deactivate()
            except fysom.Canceled:
                pass
        QtCore.QCoreApplication.instance().processEvents()  # ToDo: Is this still needed?

        # Disconnect modules from this module
        self._disconnect()

        self.__last_state = self.state
        self.sigStateChanged.emit(self.base, self.name, self.__last_state)
        self.sigAppDataChanged.emit(self.base, self.name, self.has_app_data)

        # Raise exception if by some reason no exception propagated to here and the deactivation
        # is still unsuccessful.
        if self.is_active:
            raise RuntimeError(
                f'Failed to deactivate {self.base} module "{self.name}" from {module_loc}'
            )

    @QtCore.Slot()
    def reload(self):
        # Switch to the main thread if this method was called from another thread
        if QtCore.QThread.currentThread() is not self.thread():
            QtCore.QMetaObject.invokeMethod(self, 'reload', QtCore.Qt.BlockingQueuedConnection)
            return

        with self._lock:
            # Deactivate if active
            was_active = self.is_active
            mod_to_activate = None
            if was_active:
                mod_to_activate = self.ranking_active_dependent_modules
                self._deactivate()

            # reload module
            self._load(reload=True)

            # re-activate all modules that have been active before
            if was_active:
                if mod_to_activate:
                    for module_ref in mod_to_activate:
                        module = module_ref()
                        if module is None:
                            continue
                        module.activate()
                else:
                    self._activate()

    def _load(self, reload=False):
        """
        """
        try:
            # Do nothing if already loaded and no reload is requested
            if self.is_loaded and not reload:
                return

            if self.is_remote:
                try:
                    self._instance = get_remote_module_instance(self.remote_url,
                                                                certfile=self._certfile,
                                                                keyfile=self._keyfile)
                except BaseException as e:
                    self._instance = None
                    raise RuntimeError(f'Error during initialization of remote {self.base} module '
                                       f'"{self.name}" from {self.remote_url}') from e
            else:
                # qudi module import and reload
                mod = importlib.import_module(f'qudi.{self.base}.{self.module}')
                if reload:
                    importlib.reload(mod)

                # Try getting qudi module class from imported module
                mod_class = getattr(mod, self.class_name, None)
                if mod_class is None:
                    raise AttributeError(f'No module class "{self.class_name}" found in module '
                                         f'"qudi.{self.base}.{self.module}"')

                # Check if imported class is a valid qudi module class
                if not issubclass(mod_class, Base):
                    raise TypeError(f'Qudi module class "{mod_class}" is no subclass of '
                                    f'"qudi.core.module.Base"')

                # Try to instantiate the imported qudi module class
                try:
                    self._instance = mod_class(qudi_main_weakref=self._qudi_main_ref,
                                               name=self.name,
                                               config=self.options)
                except BaseException as e:
                    self._instance = None
                    raise RuntimeError(
                        f'Error during initialization of qudi module "{self.name}" from '
                        f'"qudi.{self.base}.{self.module_name}.{self.class_name}"'
                    )
        finally:
            self.__last_state = self.state
            self.sigStateChanged.emit(self.base, self.name, self.__last_state)

    def _connect(self):
        # Check if module has already been loaded/instantiated
        if not self.is_loaded:
            raise RuntimeError(
                f'Connection failed. No module instance found for {self.base} module "{self.name}"'
            )

        # Collect all module instances required by connector config
        module_instances = {
            module_ref().name: module_ref().instance for module_ref in self.required_modules
        }
        module_connections = {conn_name: module_instances[mod_name] for conn_name, mod_name in
                              self._connections.items()}

        # Apply module connections
        self._instance.connect_modules(module_connections)

    def _disconnect(self):
        self._instance.disconnect_modules()


class _ModuleManagerMappingInterface:
    """
    """
    def __len__(self):
        with self._lock:
            return len(self._modules)

    def __getitem__(self, key):
        with self._lock:
            return self._modules[key]

    def __setitem__(self, key, value):
        if value.name != key:
            raise NameError('ManagedModule name property does not match key')
        self.add_module(value, allow_overwrite=True)

    def __delitem__(self, key):
        self.remove_module(key)

    def __iter__(self):
        with self._lock:
            for key in self._modules:
                yield key

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._modules


class ModuleManager(_ModuleManagerMappingInterface, QtCore.QObject):
    """
    """
    _instance = None  # Only class instance created will be stored here as weakref
    _lock = Mutex()

    sigModuleStateChanged = QtCore.Signal(str, str, str)
    sigModuleAppDataChanged = QtCore.Signal(str, str, bool)
    sigManagedModulesChanged = QtCore.Signal(dict)

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None or cls._instance() is None:
                obj = super().__new__(cls, *args, **kwargs)
                cls._instance = weakref.ref(obj)
                return obj
            raise RuntimeError(
                'ModuleManager is a singleton. An instance has already been created in this '
                'process. Please use ModuleManager.instance() instead.'
            )

    def __init__(self, *args, qudi_main, **kwargs):
        super().__init__(*args, **kwargs)
        self._qudi_main_ref = weakref.ref(qudi_main, self._qudi_main_ref_dead_callback)
        self._modules = dict()

    @classmethod
    def instance(cls):
        with cls._lock:
            try:
                return cls._instance()
            except TypeError:
                return None

    @property
    def module_names(self) -> List[str]:
        with self._lock:
            return list(self._modules)

    @property
    def module_states(self) -> Dict[str, str]:
        with self._lock:
            return {name: mod.state for name, mod in self._modules.items()}

    @property
    def module_instances(self) -> Dict[str, ManagedModule]:
        with self._lock:
            return {name: mod.instance for name, mod in self._modules.items() if
                    mod.instance is not None}

    @property
    def modules(self):
        return self._modules.copy()

    def remove_module(self, module_name, ignore_missing=False, emit_change=True):
        with self._lock:
            module = self._modules.pop(module_name, None)
            if module is None and not ignore_missing:
                raise KeyError(f'No module with name "{module_name}" registered.')
            module.deactivate()
            module.sigStateChanged.disconnect(self.sigModuleStateChanged)
            module.sigAppDataChanged.disconnect(self.sigModuleAppDataChanged)
            if module.allow_remote_access:
                remote_modules_server = self._qudi_main_ref().remote_modules_server
                if remote_modules_server is not None:
                    remote_modules_server.remove_shared_module(module_name)
            self.refresh_module_links()
            if emit_change:
                self.sigManagedModulesChanged.emit(self.modules)

    def add_module(self, name, base, configuration, allow_overwrite=False, emit_change=True):
        with self._lock:
            if not isinstance(name, str) or not name:
                raise TypeError('module name must be non-empty str type')
            if base not in ('gui', 'logic', 'hardware'):
                raise ValueError(f'No valid module base "{base}". '
                                 f'Unable to create qudi module "{name}".')
            if allow_overwrite:
                self.remove_module(name, ignore_missing=True)
            elif name in self._modules:
                raise ValueError(f'Module with name "{name}" already registered.')
            module = ManagedModule(self._qudi_main_ref, name, base, configuration)
            module.sigStateChanged.connect(self.sigModuleStateChanged)
            module.sigAppDataChanged.connect(self.sigModuleAppDataChanged)
            self._modules[name] = module
            self.refresh_module_links()
            # Register module in remote module service if module should be shared
            if module.allow_remote:
                remote_modules_server = self._qudi_main_ref().remote_modules_server
                if remote_modules_server is None:
                    raise RuntimeError(
                        f'Unable to share qudi module "{module.name}" as remote module. No remote '
                        f'module server running in this qudi process.'
                    )
                else:
                    logger.info(
                        f'Start sharing qudi module "{module.name}" via remote module server.'
                    )
                    remote_modules_server.share_module(module)
            if emit_change:
                self.sigManagedModulesChanged.emit(self.modules)

    def refresh_module_links(self):
        with self._lock:
            weak_refs = {
                name: weakref.ref(mod, partial(self._module_ref_dead_callback, module_name=name))
                for name, mod in self._modules.items()
            }
            for module_name, module in self._modules.items():
                # Add required module references
                required = set(module.connections.values())
                module.required_modules = set(
                    mod_ref for name, mod_ref in weak_refs.items() if name in required)
                # Add dependent module references
                module.dependent_modules = set(mod_ref for mod_ref in weak_refs.values() if
                                               module_name in mod_ref().connections.values())

    def activate_module(self, module_name):
        with self._lock:
            if module_name not in self._modules:
                raise KeyError(f'No module named "{module_name}" found in managed qudi modules. '
                               f'Module activation aborted.')
            self._modules[module_name].activate()

    def deactivate_module(self, module_name):
        with self._lock:
            if module_name not in self._modules:
                raise KeyError(f'No module named "{module_name}" found in managed qudi modules. '
                               f'Module deactivation aborted.')
            self._modules[module_name].deactivate()

    def reload_module(self, module_name):
        with self._lock:
            if module_name not in self._modules:
                raise KeyError(f'No module named "{module_name}" found in managed qudi modules. '
                               f'Module reload aborted.')
            return self._modules[module_name].reload()

    def clear_module_app_data(self, module_name):
        with self._lock:
            if module_name not in self._modules:
                raise KeyError(f'No module named "{module_name}" found in managed qudi modules. '
                               f'Can not clear module app status.')
            return self._modules[module_name].clear_app_data()

    def has_app_data(self, module_name):
        with self._lock:
            if module_name not in self._modules:
                raise KeyError(f'No module named "{module_name}" found in managed qudi modules. '
                               f'Can not check for app status file.')
            return self._modules[module_name].has_app_data()

    def start_all_modules(self):
        with self._lock:
            for module in self._modules.values():
                module.activate()

    def stop_all_modules(self):
        with self._lock:
            for module in self._modules.values():
                module.deactivate()

    def _module_ref_dead_callback(self, dead_ref, module_name):
        self.remove_module(module_name, ignore_missing=True)

    def _qudi_main_ref_dead_callback(self):
        logger.error('Qudi main reference no longer valid. This should never happen. Tearing down '
                     'ModuleManager.')
        self.clear()
