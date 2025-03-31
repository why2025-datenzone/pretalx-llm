import logging

import redis_lock
from django.core.cache import caches

logger = logging.getLogger(__name__)


class PretalxOptionalLock:
    """
    This is something rather unusual for computer science, a lock based on Redis that is allowed to fail silently.

    Some things are expensive to compute but the result can be cached. Should computing a result twice or even more often just increase the costs, but not result in any kind of data corruption, then this lock can be used for that.

    The idea here is obtain this lock when a result can't be found yet in the cache. Then with the lock, the cache is checked again and when the result is still not in the cache, it is actually computed and then placed in the cache. Then the lock is released.

    Should another process or thread try to do the same concurrently, then only one of them will get the lock and the other one will have to wait for the lock. After the first one releases the lock, the second one will then obtain the lock and find the result in the cache, therefore avoiding computing the result again.

    Should the redis server be down then the second process will get the lock immediately. Should the first process take too long or should it be killed while holding the lock, then the lock will become available again after a while for the second process. Also should there be no Redis server configured, then this will also not lock anything.
    """

    def __init__(self, lockname):
        self.lockname = lockname
        self.locked = False

    def __enter__(self):
        logger.debug("Caches is {}".format(caches))

        if "redis" not in caches:
            return
        redis_cache = caches["redis"]
        conn = redis_cache._cache.get_client(write=True)
        self.lock = redis_lock.Lock(conn, self.lockname, expire=600)
        self.locked = self.lock.acquire(timeout=300)

    def __exit__(self, type, value, traceback):
        if self.locked:
            self.lock.release()
