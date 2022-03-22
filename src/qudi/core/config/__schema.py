# -*- coding: utf-8 -*-

"""
JSON schema to be used by jsonschema.validate on YAML qudi configuration files.

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

from qudi.util.paths import get_default_data_dir

__all__ = ['qudi_cfg_schema']

qudi_cfg_schema = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'global': {
            'type': 'object',
            'additionalProperties': True,
            'default': dict(),
            'properties': {
                'startup_modules': {
                    'type': 'array',
                    'uniqueItems': True,
                    'items': {
                        'type': 'string'
                    },
                    'default': list()
                },
                'remote_modules_server': {
                    'type': ['null', 'object'],
                    'required': ['address', 'port'],
                    'default': None,
                    'additionalProperties': False,
                    'properties': {
                        'address': {
                            'type': 'string',
                        },
                        'port': {
                            'type': 'integer',
                        },
                        'certfile': {
                            'type': ['null', 'string'],
                            'default': None
                        },
                        'keyfile': {
                            'type': ['null', 'string'],
                            'default': None
                        }
                    }
                },
                'namespace_server_port': {
                    'type': 'integer',
                    'default': 18861
                },
                'force_remote_calls_by_value': {
                    'type': 'boolean',
                    'default': False
                },
                'hide_manager_window': {
                    'type': 'boolean',
                    'default': False
                },
                'stylesheet': {
                    'type': 'string',
                    'default': 'qdark.qss'
                },
                'daily_data_dirs': {
                    'type': 'boolean',
                    'default': True
                },
                'default_data_dir': {
                    'type': 'string',
                    'default': get_default_data_dir(create_missing=False)
                },
                'extension_paths': {
                    'type': 'array',
                    'uniqueItems': True,
                    'items': {
                        'type': 'string'
                    },
                    'default': list()
                }
            }
        },
        'gui': {
            'type': 'object',
            'additionalProperties': {
                '$ref': '#/$defs/local_module'
            },
            'default': dict()
        },
        'logic': {
            'type': 'object',
            'additionalProperties': {
                'oneOf': [
                    {'$ref': '#/$defs/local_module'},
                    {'$ref': '#/$defs/remote_module'}
                ]
            },
            'default': dict()
        },
        'hardware': {
            'type': 'object',
            'additionalProperties': {
                'oneOf': [
                    {'$ref': '#/$defs/local_module'},
                    {'$ref': '#/$defs/remote_module'}
                ]
            },
            'default': dict()
        }
    },

    '$defs': {
        'local_module': {
            'type': 'object',
            'required': ['module.Class'],
            'additionalProperties': False,
            'properties': {
                'module.Class': {
                    'type': 'string',
                    'pattern': r'^\w+(\.\w+)*$',
                },
                'allow_remote': {
                    'type': 'boolean',
                    'default': False
                },
                'connect': {
                    'type': 'object',
                    'additionalProperties': {
                        'type': 'string'
                    },
                    'default': dict()
                },
                'options': {
                    'type': 'object',
                    'additionalProperties': True,
                    'default': dict()
                }
            }
        },
        'remote_module': {
            'type': 'object',
            'required': ['remote_url'],
            'additionalProperties': False,
            'properties': {
                'remote_url': {
                    'type': 'string'
                },
                'certfile': {
                    'type': ['null', 'string'],
                    'default': None
                },
                'keyfile': {
                    'type': ['null', 'string'],
                    'default': None
                }
            }
        }
    }
}