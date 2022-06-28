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

from abc import abstractmethod
from typing import FrozenSet, Iterable, Optional, Any, Union, Mapping, Dict, List, Set
from functools import partial
from PySide2 import QtCore

from qudi.util.mutex import Mutex   # provides access serialization between threads
from qudi.core.logger import get_logger
from qudi.core.threadmanager import ThreadManager
from qudi.core.servers import get_remote_module_instance, get_remote_module_state
from qudi.core.servers import activate_remote_module, deactivate_remote_module, reload_remote_module
from qudi.core.servers import remote_module_has_app_data, clear_remote_module_app_data
from qudi.core.servers import toggle_remote_module_lock
from qudi.core.module import Base, ModuleStateMachine
from qudi.core.meta import ABCQObjectMeta
from qudi.util.paths import get_module_app_data_path

logger = get_logger(__name__)


class ManagedModule(QtCore.QObject, metaclass=ABCQObjectMeta):
    """ Object representing a qudi module (gui, logic or hardware) to be managed by the qudi Manager
     object. Contains status properties and handles initialization, state transitions and
     connection of the module.
    """
    sigStateChanged = QtCore.Signal(object, str)  # self, state
    sigAppDataChanged = QtCore.Signal(object, bool)  # self, has_appdata

    _managed_modules = weakref.WeakValueDictionary()

    def __new__(cls,
                name: str,
                base: str,
                config: Mapping[str, Any],
                qudi_main: object,
                module_manager: object,
                parent: Optional[QtCore.QObject] = None
                ) -> object:
        if not name or not isinstance(name, str):
            raise ValueError('Module name must be a non-empty string.')
        if base not in ['gui', 'logic', 'hardware']:
            raise ValueError('Module base must be one of ["gui", "logic", "hardware"].')
        if QtCore.QThread.currentThread() is not QtCore.QCoreApplication.instance().thread():
            raise RuntimeError('ManagedModules can only be owned by the application main thread.')
        if name in cls._managed_modules:
            raise ValueError(f'ManagedModules must have unique names. "{name}" already present.')
        obj = super().__new__(cls,
                              name=name,
                              base=base,
                              config=config,
                              qudi_main=qudi_main,
                              module_manager=module_manager,
                              parent=parent)
        cls._managed_modules[name] = obj
        return obj

    def __init__(self,
                 name: str,
                 base: str,
                 config: Mapping[str, Any],
                 qudi_main: object,
                 module_manager: object,
                 parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent=parent)

        self._name = name  # Each qudi module needs a unique string identifier
        self._base = base  # Remember qudi module base
        self._config = config  # Save reference to config
        # Create weak references to qudi main and module manager objects
        self._qudi_main_ref = weakref.ref(qudi_main)
        self._module_manager_ref = weakref.ref(module_manager)

        self._required_modules = set()
        self._dependent_modules = set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def base(self) -> str:
        return self._base

    @property
    def required_module_names(self) -> Set[str]:
        return {mod.name for mod in self._required_modules}

    @property
    def dependent_module_names(self) -> Set[str]:
        return {mod.name for mod in self._dependent_modules}

    @property
    def is_loaded(self) -> bool:
        return self.state != 'not loaded'

    @property
    def is_active(self) -> bool:
        return self.state not in ['not loaded', 'deactivated']

    @property
    def is_busy(self) -> bool:
        return self.state == 'busy'

    def register_dependent_module(self, module: object) -> None:
        if self.is_active:
            self._dependent_modules.add(module)
        else:
            raise RuntimeError(f'Tried to register dependent module "{module.name}" on '
                               f'non-active module "{self.name}"')

    def unregister_dependent_module(self, module: object) -> None:
        self._dependent_modules.discard(module)

    def get_ranking_dependent_modules(self) -> set:
        ranking_dependent_modules = set()
        for module in self._dependent_modules:
            if module.is_active:
                ranking_modules = module.get_ranking_dependent_modules()
                if ranking_modules:
                    ranking_dependent_modules.update(ranking_modules)
                else:
                    ranking_dependent_modules.add(module)
        return ranking_dependent_modules

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def reload(self) -> None:
        pass

    @abstractmethod
    def activate(self) -> None:
        pass

    @abstractmethod
    def deactivate(self) -> None:
        pass

    @abstractmethod
    def lock(self) -> None:
        pass
        # if self._mutex.try_lock():
        #     try:
        #         self._lock()
        #     finally:
        #         self._mutex.unlock()
        #         self.sigStateChanged.emit(self, self.state)

    @abstractmethod
    def unlock(self) -> None:
        pass

    @abstractmethod
    def clear_app_data(self) -> None:
        pass

    @property
    @abstractmethod
    def has_app_data(self) -> bool:
        pass

    @property
    @abstractmethod
    def instance(self) -> Union[None, Base]:
        pass

    @property
    @abstractmethod
    def state(self) -> str:
        pass


class LocalManagedModule(ManagedModule):
    """ Local qudi module
    """

    def __init__(self,
                 name: str,
                 base: str,
                 config: Mapping[str, Any],
                 qudi_main: object,
                 module_manager: object,
                 parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(name=name,
                         base=base,
                         config=config,
                         qudi_main=qudi_main,
                         module_manager=module_manager,
                         parent=parent)
        # Extract module and class name
        self._module_name, self._class_name = self._config['module.Class'].rsplit('.', 1)
        # Remember connections by name
        self._connections = self._config['connect']
        # See if remote access to this module is allowed
        self._allow_remote = self._config['allow_remote']
        # Get module options
        self._options = self._config['options']
        # Reference to created module instance (after loading)
        self._instance = None
        self._status_file_path = get_module_app_data_path(self._class_name,
                                                          self._module_name,
                                                          self.name)
        self._thread_name = f'mod-{self.base}-{self.name}'

    @property
    def instance(self) -> Union[None, Base]:
        return self._instance

    @property
    def has_app_data(self) -> bool:
        return os.path.exists(self._status_file_path)

    @property
    def state(self) -> str:
        try:
            return self._instance.module_state()
        except AttributeError:
            return 'not loaded'

    def load(self, reload: Optional[bool] = False) -> None:
        # Do nothing if already loaded
        if self.is_loaded:
            return

        try:
            # qudi module import
            mod = importlib.import_module(f'qudi.{self.base}.{self._module_name}')
            if reload:
                mod = importlib.reload(mod)

            # Try getting qudi module class from imported module
            mod_class = getattr(mod, self._class_name, None)
            if mod_class is None:
                raise AttributeError(
                    f'Could not import "{self._class_name}" from "qudi.{self.base}.{self._module_name}"'
                )

            # Check if imported class is a valid qudi module class
            if not issubclass(mod_class, Base):
                raise TypeError(
                    f'Qudi module class "qudi.{self.base}.{self._module_name}.{self._class_name}" is '
                    f'no subclass of "{Base.__module__}.{Base.__name__}"'
                )

            # Try to instantiate the imported qudi module class
            try:
                self._instance = mod_class(qudi_main_weakref=self._qudi_main_ref,
                                           name=self.name,
                                           config=self._options)
                self._required_modules.clear()
                self._instance.module_state.sigStateChanged.connect(self._state_change_callback,
                                                                    QtCore.Qt.QueuedConnection)
            except BaseException as err:
                self._instance = None
                raise RuntimeError(
                    f'Exception during initialization of qudi module "{self.name}" from '
                    f'"qudi.{self.base}.{self._module_name}.{self._class_name}"'
                ) from err
        finally:
            self.sigStateChanged.emit(self, self.state)

    def reload(self) -> None:
        # Do a normal load if it was not already loaded
        if not self.is_loaded:
            return self.load()

        # Deactivate if active
        was_active = self.is_active
        if was_active:
            modules_to_activate = self.get_ranking_dependent_modules()
            self.deactivate()
        else:
            modules_to_activate = set()

        # reload module
        self._instance.module_state.sigStateChanged.disconnect(self._state_change_callback)
        self._instance = None
        self.load(reload=True)

        # re-activate all modules that have been active before
        if was_active:
            self.activate()
            for module in modules_to_activate:
                module.activate()

    def activate(self) -> None:
        # Return early if already active
        if self.is_active:
            # If it is a GUI module, show it again.
            if self.base == 'gui':
                self._instance.show()
            return

        # Load first if not already loaded
        self.load()

        # Recursive activation of required modules
        modules_to_connect = dict()
        for connector_name, module_name in self._connections.items():
            try:
                modules_to_connect[connector_name] = self._managed_modules[module_name]
            except KeyError:
                continue
            modules_to_connect[connector_name].activate()

        logger.info(
            f'Activating {self.base} module "{self.name}" from '
            f'{self._module_name}.{self._class_name}'
        )

        try:
            # Establish module interconnections via Connector meta object in qudi module instance
            self._instance.connect_modules(
                {conn: mod.instance for conn, mod in modules_to_connect.items()}
            )

            # Activate this module
            if self._instance.module_threaded:
                thread_manager = ThreadManager.instance()
                if thread_manager is None:
                    raise RuntimeError('No ThreadManager instantiated in current qudi process. '
                                       'Unable to activate threaded modules.')
                thread = thread_manager.get_new_thread(self._thread_name)
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
                        thread_manager.quit_thread(self._thread_name)
                        thread_manager.join_thread(self._thread_name)
            else:
                self._instance.module_state.activate()

            if self.is_active:
                # Register dependency in connected modules
                for module in modules_to_connect.values():
                    module.register_dependent_module(self)
                    self._required_modules.add(module)
            else:
                # Raise exception if by some reason no exception propagated to here and the
                # activation is still unsuccessful.
                try:
                    self._instance.disconnect_modules()
                except:
                    pass
                raise RuntimeError(
                    f'Failed to activate {self.base} module "{self.name}" from {self._module_name}.'
                    f'{self._class_name}'
                )
        finally:
            self.sigStateChanged.emit(self, self.state)
            self.sigAppDataChanged.emit(self, self.has_app_data)

    def deactivate(self) -> None:
        # Return early if nothing to do
        if not self.is_active:
            return

        # Recursively deactivate dependent modules. Create a list-copy of the set first.
        for module in list(self._dependent_modules):
            module.deactivate()

        logger.info(f'Deactivating {self.base} module "{self.name}" from '
                    f'{self._module_name}.{self._class_name}')

        # Actual deactivation of this module
        try:
            if self._instance.module_threaded:
                thread_manager = ThreadManager.instance()
                if thread_manager is None:
                    raise RuntimeError('No ThreadManager instantiated in current qudi process. '
                                       'Unable to properly deactivate threaded modules.')
                try:
                    QtCore.QMetaObject.invokeMethod(self._instance.module_state,
                                                    'deactivate',
                                                    QtCore.Qt.BlockingQueuedConnection)
                finally:
                    QtCore.QMetaObject.invokeMethod(self._instance,
                                                    'move_to_main_thread',
                                                    QtCore.Qt.BlockingQueuedConnection)
                    thread_manager.quit_thread(self._thread_name)
                    thread_manager.join_thread(self._thread_name)
            else:
                self._instance.module_state.deactivate()

            QtCore.QCoreApplication.instance().processEvents()  # ToDo: Is this still needed?

            # Disconnect modules from this module
            self._instance.disconnect_modules()

            # Raise exception if by some reason no exception propagated to here and the deactivation
            # is still unsuccessful.
            if self.is_active:
                raise RuntimeError(
                    f'Failed to deactivate {self.base} module "{self.name}" from '
                    f'{self._module_name}.{self._class_name}'
                )
        finally:
            self.sigStateChanged.emit(self, self.state)
            self.sigAppDataChanged.emit(self, self.has_app_data)

    def lock(self) -> None:
        if self._instance is None:
            raise RuntimeError(f'Unable to lock module "{self.name}". Module is not loaded yet.')
        self._instance.module_state.lock()

    def unlock(self) -> None:
        if self._instance is None:
            raise RuntimeError(f'Unable to unlock module "{self.name}". Module is not loaded yet.')
        self._instance.module_state.unlock()

    def clear_app_data(self) -> None:
        try:
            os.remove(self._status_file_path)
        except FileNotFoundError:
            pass
        finally:
            self.sigAppDataChanged.emit(self, self.has_app_data)


class RemoteManagedModule(ManagedModule):
    """ Remote qudi module
    """
    def __init__(self,
                 name: str,
                 base: str,
                 config: Mapping[str, Any],
                 qudi_main: object,
                 module_manager: object,
                 parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(name=name,
                         base=base,
                         config=config,
                         qudi_main=qudi_main,
                         module_manager=module_manager,
                         parent=parent)
        # Extract rpyc URL
        self._url = self._config['remote_url']
        # Remember connections by name
        self._certfile = self._config['certfile']
        # See if remote access to this module is allowed
        self._keyfile = self._config['keyfile']
        # Remember last state to detect state changes
        self._last_state = 'not loaded'

        if not self._certfile:
            self._certfile = None
        if not self._keyfile:
            self._keyfile = None
        if self._certfile is None or self._keyfile is None:
            self._certfile = None
            self._keyfile = None

    @property
    def instance(self) -> Union[None, Base]:
        return get_remote_module_instance(self._url, certfile=self._certfile, keyfile=self._keyfile)

    @property
    def has_app_data(self) -> bool:
        return remote_module_has_app_data(self._url, certfile=self._certfile, keyfile=self._keyfile)

    @property
    def state(self) -> str:
        state = get_remote_module_state(self._url, certfile=self._certfile, keyfile=self._keyfile)
        if not state:
            raise RuntimeError(
                f'Unable to retrieve module state from remote module "{self.name}" ({self._url})'
            )
        return state

    def load(self) -> None:
        # Do nothing if already loaded
        if self.is_loaded:
            return

        try:
            instance = get_remote_module_instance(self._url,
                                                  certfile=self._certfile,
                                                  keyfile=self._keyfile)
            if instance is None:
                raise RuntimeError(f'Unable to load remote module "{self.name}" ({self._url})')
        finally:
            self.sigStateChanged.emit(self, self.state)

    def reload(self) -> None:
        # Do a normal load if it was not already loaded
        if not self.is_loaded:
            return self.load()

        try:
            # Deactivate if active
            was_active = self.is_active
            if was_active:
                modules_to_activate = self.get_ranking_dependent_modules()
            else:
                modules_to_activate = set()

            # reload module
            if not reload_remote_module(self._url, certfile=self._certfile, keyfile=self._keyfile):
                raise RuntimeError(f'Unable to reload remote module "{self.name}" ({self._url})')

            # re-activate all modules that have been active before
            if was_active:
                for module in modules_to_activate:
                    module.activate()
        finally:
            self.sigStateChanged.emit(self, self.state)

    def activate(self) -> None:
        # Return early if already active
        if self.is_active:
            return

        try:
            # Load first if not already loaded
            self.load()

            logger.info(f'Activating remote {self.base} module "{self.name}" at {self._url}')

            # Activate this module
            if not activate_remote_module(self._url, certfile=self._certfile, keyfile=self._keyfile):
                raise RuntimeError(f'Unable to activate remote module "{self.name}" ({self._url})')
        finally:
            self.sigStateChanged.emit(self, self.state)
            self.sigAppDataChanged.emit(self, self.has_app_data)

    def deactivate(self) -> None:
        # Return early if nothing to do
        if not self.is_active:
            return

        try:
            # Recursively deactivate dependent modules. Create a list-copy of the set first.
            for module in list(self._dependent_modules):
                module.deactivate()

            logger.info(f'Deactivating remote {self.base} module "{self.name}" at {self._url}')

            # Actual deactivation of this module
            if not deactivate_remote_module(self._url, certfile=self._certfile, keyfile=self._keyfile):
                raise RuntimeError(
                    f'Unable to deactivate remote module "{self.name}" ({self._url})'
                )
        finally:
            self.sigStateChanged.emit(self, self.state)
            self.sigAppDataChanged.emit(self, self.has_app_data)

    def lock(self) -> None:
        try:
            if not toggle_remote_module_lock(self._url,
                                             True,
                                             certfile=self._certfile,
                                             keyfile=self._keyfile):
                raise RuntimeError(f'Unable to lock remote module "{self.name}" ({self._url})')
        finally:
            self.sigStateChanged.emit(self, self.state)

    def unlock(self) -> None:
        try:
            if not toggle_remote_module_lock(self._url,
                                             False,
                                             certfile=self._certfile,
                                             keyfile=self._keyfile):
                raise RuntimeError(f'Unable to unlock remote module "{self.name}" ({self._url})')
        finally:
            self.sigStateChanged.emit(self, self.state)

    def clear_app_data(self) -> None:
        try:
            if not clear_remote_module_app_data(self._url,
                                                certfile=self._certfile,
                                                keyfile=self._keyfile):
                raise RuntimeError(
                    f'Unable to clear AppData of remote module "{self.name}" ({self._url})'
                )
        finally:
            self.sigAppDataChanged.emit(self, self.has_app_data)


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

    def clear(self) -> None:
        with self._lock:
            for mod_name in list(self._modules):
                self._remove_module(mod_name, ignore_missing=True)
            self.sigManagedModulesChanged.emit(self.modules)


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

    def remove_module(self, module_name: str, ignore_missing: Optional[bool] = False) -> None:
        with self._lock:
            self._remove_module(module_name, ignore_missing)
            self.sigManagedModulesChanged.emit(self.modules)

    def _remove_module(self, module_name: str, ignore_missing: Optional[bool] = False) -> None:
        module = self._modules.pop(module_name, None)
        if module is None and not ignore_missing:
            raise KeyError(f'No module with name "{module_name}" registered.')
        module.deactivate()
        module.sigStateChanged.disconnect(self._module_state_change_callback)
        module.sigAppDataChanged.disconnect(self._module_appdata_change_callback)
        if module.allow_remote_access:
            remote_modules_server = self._qudi_main_ref().remote_modules_server
            if remote_modules_server is not None:
                remote_modules_server.remove_shared_module(module_name)
        self._refresh_module_links()

    def add_module(self,
                   name: str,
                   base: str,
                   configuration: Mapping[str, Any],
                   allow_overwrite: Optional[bool] = False
                   ) -> None:
        with self._lock:
            self._add_module(name, base, configuration, allow_overwrite)
            self.sigManagedModulesChanged.emit(self.modules)

    def _add_module(self,
                    name: str,
                    base: str,
                    configuration:
                    Mapping[str, Any],
                    allow_overwrite: Optional[bool] = False
                    ) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError('module name must be non-empty str type')
        if base not in ('gui', 'logic', 'hardware'):
            raise ValueError(f'No valid module base "{base}". '
                             f'Unable to create qudi module "{name}".')
        if allow_overwrite:
            self._remove_module(name, ignore_missing=True)
        elif name in self._modules:
            raise ValueError(f'Module with name "{name}" already registered.')
        module = ManagedModule(name, base, configuration, self._qudi_main_ref)
        module.sigStateChanged.connect(self._module_state_change_callback)
        module.sigAppDataChanged.connect(self._module_appdata_change_callback)
        self._modules[name] = module
        self._refresh_module_links()
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

    def _refresh_module_links(self):
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
        module = self._modules.get(module_name, None)
        if module is None:
            raise KeyError(f'No module named "{module_name}" found in managed qudi modules. '
                           f'Can not check for app status file.')
        return module.has_app_data()

    def start_all_modules(self):
        with self._lock:
            for module in self._modules.values():
                module.activate()

    def stop_all_modules(self):
        with self._lock:
            for module in self._modules.values():
                module.deactivate()

    @QtCore.Slot(object)
    def _module_state_change_callback(self, module: ManagedModule) -> None:
        self.sigModuleStateChanged.emit(module.base, module.name, module.state)

    @QtCore.Slot(object)
    def _module_appdata_change_callback(self, module: ManagedModule) -> None:
        self.sigModuleAppDataChanged.emit(module.base, module.name, module.has_app_data)

    def _module_ref_dead_callback(self, dead_ref, module_name):
        self.remove_module(module_name, ignore_missing=True)

    def _qudi_main_ref_dead_callback(self):
        logger.error('Qudi main reference no longer valid. This should never happen. Tearing down '
                     'ModuleManager.')
        self.clear()
