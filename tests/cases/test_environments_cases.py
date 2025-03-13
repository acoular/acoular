# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Test cases for all environments and flow fields."""

import acoular as ac
from pytest_cases import parametrize

from tests.utils import get_subclasses

ENV_SKIP_DEFAULT = [
    ac.UniformFlowEnvironment,
    ac.GeneralFlowEnvironment,
]
FLOW_SKIP_DEFAULT = [
    ac.FlowField,
    ac.SlotJet,
    ac.OpenJet,
    ac.RotatingFlow,
]

ENV_DEFAULT = [env for env in get_subclasses(ac.Environment) if env not in ENV_SKIP_DEFAULT]
FLOW_DEFAULT = [flow for flow in get_subclasses(ac.FlowField) if flow not in FLOW_SKIP_DEFAULT]


class Flows:
    """Test cases for all flow fields.

    New flow fields should be added here. If no dedicated test case is added for a
    :class:`FlowField` derived class, the class is still included in the test suite through the use
    of the `case_default` case. If a dedicated test case was added for a flow field, it should be
    added to the `FLOW_SKIP_DEFAULT` list, which excludes the class from `case_default`.

    New cases should create an instance of the flow field with a specific parameterization and
    return it.
    """

    @parametrize('flow', FLOW_DEFAULT)
    def case_default(self, flow):
        return flow()

    def case_SlotJet(self):
        return ac.SlotJet(v0=70.0, origin=(-0.7, 0, 0.7))

    def case_OpenJet(self):
        return ac.OpenJet(v0=70.0, origin=(-0.7, 0, 0.7))

    def case_RotatingFlow(self):
        return ac.RotatingFlow(v0=70.0, rps=-1000.0 / 60)


class Environments:
    """Test cases for all environments.

    New environments should be added here. If no dedicated test case is added for a
    :class:`Environment` derived class, the class is still included in the test suite through the
    use of the `case_default` case. If a dedicated test case was added for an environment, it should
    be added to the `ENV_SKIP_DEFAULT` list, which excludes the class from `case_default`.

    New cases should create an instance of the environment with a specific parameterization and
    return it.
    """

    @parametrize('env', ENV_DEFAULT)
    def case_default(self, env):
        return env()

    def case_UniformFlowEnvironment(self):
        return ac.UniformFlowEnvironment(ma=0.3)

    def case_GeneralFlowEnvironment(self):
        return ac.GeneralFlowEnvironment(ff=ac.OpenJet(v0=70.0, origin=(-0.7, 0, 0.7)))
