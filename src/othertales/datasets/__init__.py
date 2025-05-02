"""Dataset Creator Agent Package"""

# This is a namespace package
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)

# Import the components directly - do not use relative imports in __init__.py
# This prevents circular import errors when running as a module
