import numpy as np

from memory.pattern import PatternNode


def test_pattern_node_update_and_branch():
    node = PatternNode(key=(1,))
    assert node.count == 0
    hs = np.ones(8)
    node.update(reward=1.0, error=0.5, hidden_state=hs)
    assert node.count == 1
    assert node.hidden_states_history
    var = node.get_reward_variance()
    assert isinstance(var, float)

    child = node.create_branch('test')
    assert 'test' in node.children
    assert child.metadata.get('parent') == node.key
