# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Unit tests for environments module."""

import acoular as ac
import numpy as np


class TestEnvironmentPublicAPI:
    """Test the public API of Environment classes."""

    def test_environment_apparent_r_exists(self):
        """Test that apparent_r method exists and is public."""
        env = ac.Environment()
        assert hasattr(env, 'apparent_r')
        assert callable(env.apparent_r)

    def test_environment_apparent_r_basic(self):
        """Test basic functionality of apparent_r method."""
        env = ac.Environment()
        # Single grid point, single mic position
        gpos = np.array([[1.0], [0.0], [0.0]])
        mpos = np.array([[0.0], [0.0], [0.0]])
        result = env.apparent_r(gpos, mpos)
        assert result.shape == (1,)
        np.testing.assert_allclose(result, [1.0], rtol=1e-10)

    def test_environment_apparent_r_multiple_points(self):
        """Test apparent_r with multiple grid points and microphones."""
        env = ac.Environment()
        gpos = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
        mpos = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        result = env.apparent_r(gpos, mpos)
        assert result.shape == (2, 2)
        # Check distances are positive
        assert np.all(result >= 0)

    def test_environment_apparent_r_scalar_mpos(self):
        """Test apparent_r with scalar mpos (defaults to origin)."""
        env = ac.Environment()
        gpos = np.array([[3.0], [4.0], [0.0]])
        result = env.apparent_r(gpos, 0.0)
        assert result.shape == (1,)
        np.testing.assert_allclose(result, [5.0], rtol=1e-10)

    def test_uniformflow_apparent_r_exists(self):
        """Test that apparent_r method exists in UniformFlowEnvironment."""
        env = ac.UniformFlowEnvironment(ma=0.3)
        assert hasattr(env, 'apparent_r')
        assert callable(env.apparent_r)

    def test_uniformflow_apparent_r_basic(self):
        """Test basic functionality of apparent_r in UniformFlowEnvironment."""
        env = ac.UniformFlowEnvironment(ma=0.3)
        gpos = np.array([[1.0], [0.0], [0.0]])
        mpos = np.array([[0.0], [0.0], [0.0]])
        result = env.apparent_r(gpos, mpos)
        assert result.shape == (1,)
        # With flow, apparent distance differs from physical distance
        assert result[0] > 0

    def test_generalflow_apparent_r_exists(self):
        """Test that apparent_r method exists in GeneralFlowEnvironment."""
        flow_field = ac.OpenJet(v0=70.0, origin=(-0.7, 0, 0.7))
        env = ac.GeneralFlowEnvironment(ff=flow_field)
        assert hasattr(env, 'apparent_r')
        assert callable(env.apparent_r)

    def test_environment_apparent_r_vs_dist_mat(self):
        """Test that apparent_r gives same result as dist_mat for basic Environment."""
        env = ac.Environment()
        gpos = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])
        mpos = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

        result_apparent = env.apparent_r(gpos, mpos)
        result_distmat = ac.environments.dist_mat(
            np.ascontiguousarray(gpos), np.ascontiguousarray(mpos)
        )

        np.testing.assert_allclose(result_apparent, result_distmat, rtol=1e-10)
