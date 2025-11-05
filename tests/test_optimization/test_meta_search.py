"""
Unit tests for meta_search.py (MetaParameterSearch interface).
"""
import pytest
from optimization.meta_search import MetaParameterSearch

def test_meta_parameter_search_init():
    s = MetaParameterSearch(method='pso')
    assert s.method == 'pso'
    s2 = MetaParameterSearch(method='bayesian')
    assert s2.method == 'bayesian'


def test_meta_parameter_search_invalid_method():
    with pytest.raises(ValueError):
        MetaParameterSearch(method='invalid').search([], [], None)

# Note: search() integration is not unit tested here due to external dependencies.
