"""
Simple smoke test for caching implementation.
Tests basic cache operations without heavy dependencies.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cache import Cache, make_cache_key

def test_basic_cache():
    """Test basic cache operations."""
    print("Testing basic cache operations...")
    cache = Cache()
    
    # Test set and get
    cache.set('test_key', {'value': 42})
    result = cache.get('test_key')
    assert result is not None, "Cache get failed"
    assert result['value'] == 42, "Cache value incorrect"
    print("✓ Basic cache set/get works")
    
    # Test miss
    result = cache.get('nonexistent_key')
    assert result is None, "Cache should return None for missing key"
    print("✓ Cache miss returns None")
    
    # Test stats
    stats = cache.get_stats()
    assert 'memory_entries' in stats, "Stats missing memory_entries"
    assert stats['memory_entries'] == 1, "Stats incorrect"
    print("✓ Cache stats work")
    print(f"  Stats: {stats}")
    
    return True

def test_cache_key():
    """Test cache key generation."""
    print("\nTesting cache key generation...")
    
    # Test basic keys
    key1 = make_cache_key(1, 2, 3)
    key2 = make_cache_key(1, 2, 3)
    key3 = make_cache_key(1, 2, 4)
    
    assert key1 == key2, "Same inputs should produce same key"
    assert key1 != key3, "Different inputs should produce different key"
    print("✓ Cache key generation works")
    print(f"  Sample key: {key1[:16]}...")
    
    return True

def test_cache_persistence():
    """Test cache disk persistence."""
    print("\nTesting cache persistence...")
    cache = Cache()
    
    # Set with persistence
    cache.set('persist_key', 'persisted_value', persist=True)
    
    # Create new cache instance (should load from disk)
    cache2 = Cache()
    result = cache2.get('persist_key')
    
    assert result == 'persisted_value', "Persisted value not retrieved"
    print("✓ Cache persistence works")
    
    # Cleanup
    cache.clear()
    print("✓ Cache cleanup works")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("CACHE SMOKE TESTS")
    print("=" * 60)
    
    try:
        test_basic_cache()
        test_cache_key()
        test_cache_persistence()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
