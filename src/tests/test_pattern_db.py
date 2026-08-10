from memory.pattern_db import PatternDB
from memory.pattern import PatternNode


def test_pattern_db_basic_operations():
    db = PatternDB(max_patterns=10)
    key = (1,2,3)
    node = db.get(key)
    assert isinstance(node, PatternNode)

    node.count += 1
    db.update(key, node)
    stats = db.get_stats()
    assert stats['n_patterns'] >= 1

    # prune should not remove when under limit
    removed = db.prune_least_frequent(keep_ratio=0.9)
    assert isinstance(removed, int)
    db.clear()
    assert db.get_size() == 0
