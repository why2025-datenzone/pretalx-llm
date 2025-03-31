from hashlib import sha256

from django.core.cache import caches

# Cache values, silently fail when there is no cache available


def maybe_get(key):
    if "default" in caches:
        return caches["default"].get(key)


def maybe_set(key, value, timeout=None):
    if "default" in caches:
        caches["default"].set(key, value, timeout=timeout)


def hash_string(str):
    hasher = sha256()
    hasher.update(str.encode("utf-8"))
    return hasher.hexdigest()
