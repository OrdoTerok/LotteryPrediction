"""
Unit tests for particle_swarm.py (PSO optimization).
"""
import pytest
import numpy as np
from optimization import particle_swarm


def test_check_constraints_default():
    # By default, always returns True
    assert particle_swarm.check_constraints(['a', 'b'], [1, 2]) is True


def test_get_valid_initial_value_float():
    val = particle_swarm.get_valid_initial_value('lr', 0.001, 0.1)
    assert 0.001 <= val <= 0.1
    assert isinstance(val, float)


def test_get_valid_initial_value_int():
    val = particle_swarm.get_valid_initial_value('units', 4, 32)
    assert 4 <= val <= 32
    assert isinstance(val, int)


def test_is_data_valid_empty():
    assert not particle_swarm.is_data_valid(None, None)


def test_is_data_valid_nan():
    import pandas as pd
    df = pd.DataFrame({'a': [1, np.nan]})
    assert not particle_swarm.is_data_valid(df, df)


def test_particle_init_and_update():
    bounds = [(0, 1), (0, 1)]
    var_names = ['x', 'y']
    p = particle_swarm.Particle(bounds, var_names)
    assert len(p.position) == 2
    old_position = p.position.copy()
    p.update_velocity(global_best=np.array([0.5, 0.5]))
    p.update_position(bounds)
    assert p.position.shape == (2,)
    # Should not error
    p.set_vars()

# Note: Full PSO and fitness_func tests require integration with model/config/data, so are not unit tested here.
