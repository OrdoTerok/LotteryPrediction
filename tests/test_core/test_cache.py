import os
import shutil
from core.cache import Cache

def test_cache_set_get_clear(tmp_path):
    cache_dir = tmp_path / 'cache'
    cache = Cache(str(cache_dir))
    cache.set('foo', 123)
    assert cache.get('foo') == 123
    cache.clear()
    assert cache.get('foo') is None

def test_cache_persistence(tmp_path):
    cache_dir = tmp_path / 'cache'
    cache = Cache(str(cache_dir))
    cache.set('bar', 456)
    # New instance, should load from disk
    cache2 = Cache(str(cache_dir))
    assert cache2.get('bar') == 456
    cache2.clear()
