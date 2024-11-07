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

    @parametrize('flow', FLOW_DEFAULT)
    def case_default(self, flow):
        return flow()

    def case_SlotJet(self):
        return ac.SlotJet(v0=70.0, origin=(-0.7, 0, 0.7))

    def case_OpenJet(self):
        return ac.OpenJet(v0=70.0, origin=(-0.7, 0, 0.7))

    def case_RotatingFlow(self):
        return ac.RotatingFlow(v0=70.0, rpm=1000.0)


class Environments:

    @parametrize('env', ENV_DEFAULT)
    def case_default(self, env):
        return env()

    def case_UniformFlowEnvironment(self):
        return ac.UniformFlowEnvironment(ma=0.3)

    def case_GeneralFlowEnvironment(self):
        return ac.GeneralFlowEnvironment(
            ff=ac.OpenJet(v0=70.0, origin=(-0.7, 0, 0.7)))
