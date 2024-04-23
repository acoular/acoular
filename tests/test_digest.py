from acoular import UniformFlowEnvironment,SlotJet,OpenJet,RotatingFlow,MicGeom, SteeringVector,\
    BeamformerOrth, RectGrid, MaskedTimeSamples, Sector,\
        UncorrelatedNoiseSource, SourceMixer, SamplesGenerator, BeamformerTimeTraj, BeamformerTimeSqTraj,\
            BeamformerCleantTraj, BeamformerCleantSqTraj, IntegratorSectorTime, MaskedTimeInOut, ChannelMixer,\
                SpatialInterpolator, SpatialInterpolatorRotation, SpatialInterpolatorConstantRotation, Mixer,\
                    WriteWAV, MergeGrid, FiltWNoiseGenerator, SphericalHarmonicSource, PointSource
from numpy import array
from unittest import TestCase

# a dictionary containing all classes that should change their digest on
# changes of the following trait types:
#   * List
#   * CArray 
UNEQUAL_DIGEST_TEST_DICT = {
#    "MicGeom.mpos_tot item assignment" : (MicGeom(mpos_tot=[[1.,2.,3.]]), "obj.mpos_tot[:] = 0."),
    "MicGeom.mpos_tot new array assignment" : (MicGeom(mpos_tot=[[1.,2.,3.]]), "obj.mpos_tot = array([0.])"),
#    "MicGeom.invalid_channels item assignment" : (MicGeom(mpos_tot=[[1.,2.,3.]],invalid_channels=[1]), "obj.invalid_channels[0] = 0"),
    "MicGeom.invalid_channels new list assignment" : (MicGeom(mpos_tot=[[1.,2.,3.]],invalid_channels=[1]), "obj.invalid_channels = [0]"),
    # environments.py
#    "UniformFlowEnvironment.fdv item assignment": (UniformFlowEnvironment(), "obj.fdv[0] = 0."),
    "UniformFlowEnvironment.fdv array assignment": (UniformFlowEnvironment(), "obj.fdv = array((0., 0., 0.))"),
#    "SlotJet.origin item assignment": (SlotJet(), "obj.origin[0] = 1."),
    "SlotJet.origin array assignment": (SlotJet(), "obj.origin = array((1., 0., 0.))"),
#    "SlotJet.flow item assignment": (SlotJet(), "obj.flow[0] = 0."),
    "SlotJet.flow array assignment": (SlotJet(), "obj.flow = array((0., 0., 0.))"),    
#    "SlotJet.plane item assignment": (SlotJet(), "obj.plane[0] = 1."),
    "SlotJet.plane array assignment": (SlotJet(), "obj.plane = array((1., 0., 0.))"),
#    "OpenJet.origin item assignment": (OpenJet(), "obj.origin[0] = 1."),
    "OpenJet.origin array assignment": (OpenJet(), "obj.origin = array((1., 0., 0.))"),   
#    "RotatingFlow.origin item assignment": (RotatingFlow(), "obj.origin[0] = 1."),
    "RotatingFlow.origin array assignment": (RotatingFlow(), "obj.origin = array((1., 0., 0.))"),  
    #fbeamform.py
#    "SteeringVector.ref item assignment": (SteeringVector(), "obj.ref[0] = 1."),  
    "SteeringVector.ref array assignment": (SteeringVector(), "obj.ref = array((1., 1., 1.))"),  
#    "BeamformerOrth.eva_list item assignment": (BeamformerOrth(eva_list=array((0, 1))), "obj.eva_list[0] = 2"),  
    "BeamformerOrth.eva_list array assignment": (BeamformerOrth(eva_list=array((0, 1))), "obj.eva_list = array((2))"),  
    #grids.py
    "MergeGrid.grids item assignment": (MergeGrid(grids=[RectGrid()]), "obj.grids[0] = RectGrid()"),  
    "MergeGrid.grids list assignment": (MergeGrid(), "obj.grids = [RectGrid()]"),
    # signals.py
#    "FiltWNoiseGenerator.ar item assignment": (FiltWNoiseGenerator(ar=[1.,2.,3.]), "obj.ar[0] = 0."),
    "FiltWNoiseGenerator.ar array assignment": (FiltWNoiseGenerator(ar=[1.,2.,3.]), "obj.ar = array((0., 0., 0.))"),   
#    "FiltWNoiseGenerator.ma item assignment": (FiltWNoiseGenerator(ma=[1.,2.,3.]), "obj.ma[0] = 0."),
    "FiltWNoiseGenerator.mar array assignment": (FiltWNoiseGenerator(ma=[1.,2.,3.]), "obj.ma = array((0., 0., 0.))"), 
    #sources.py 
    "MaskedTimeSamples.invalid_channels item assignment": (MaskedTimeSamples(invalid_channels=[1]), "obj.invalid_channels[0] = 0"),  
    "MaskedTimeSamples.invalid_channels list assignment": (MaskedTimeSamples(), "obj.invalid_channels = [0]"),
#    "SphericalHarmonicSource.alpha item assignment": (SphericalHarmonicSource(alpha=array((0, 1))), "obj.alpha[0] = 1."),  
    "SphericalHarmonicSource.alpha array assignment": (SphericalHarmonicSource(alpha=array((0, 1))), "obj.alpha = array((1., 1., 1.))"),  
#    "UncorrelatedNoiseSource.seed item assignment": (UncorrelatedNoiseSource(seed=array((1, 2))), "obj.seed[0] = 3"),  
    "UncorrelatedNoiseSource.seed array assignment": (UncorrelatedNoiseSource(seed=array((1, 2))), "obj.seed = array((3,4))"),  
    "SourceMixer.sources item assignment": (SourceMixer(sources=[SamplesGenerator()]), "obj.sources[0] = PointSource()"),  
    "SourceMixer.sources list assignment": (SourceMixer(sources=[SamplesGenerator()]), "obj.sources = [PointSource()]"),  
    # tbeamform.py
#    "BeamformerTimeTraj.rvec item assignment": (BeamformerTimeTraj(), "obj.rvec[0] = 1."),
    "BeamformerTimeTraj.rvec array assignment": (BeamformerTimeTraj(), "obj.rvec = array((1., 0., 0.))"),
#    "BeamformerTimeSqTraj.rvec item assignment": (BeamformerTimeSqTraj(), "obj.rvec[0] = 1."),
    "BeamformerTimeSqTraj.rvec array assignment": (BeamformerTimeSqTraj(), "obj.rvec = array((1., 0., 0.))"),
#    "BeamformerCleantTraj.rvec item assignment": (BeamformerCleantTraj(), "obj.rvec[0] = 1."),
    "BeamformerCleantTraj.rvec array assignment": (BeamformerCleantTraj(), "obj.rvec = array((1., 0., 0.))"),
#    "BeamformerCleantSqTraj.rvec item assignment": (BeamformerCleantSqTraj(), "obj.rvec[0] = 1."),
    "BeamformerCleantSqTraj.rvec array assignment": (BeamformerCleantSqTraj(), "obj.rvec = array((1., 0., 0.))"),
    "IntegratorSectorTime.sectors item assignment": (IntegratorSectorTime(sectors=[Sector()]), "obj.sectors[0] = Sector()"),  
    "IntegratorSectorTime.sectors list assignment": (IntegratorSectorTime(sectors=[Sector()]), "obj.sectors = [Sector()]"),  
    # tprocess.py
    "MaskedTimeInOut.invalid_channels item assignment": (MaskedTimeInOut(invalid_channels=[1]), "obj.invalid_channels[0] = 0"),  
    "MaskedTimeInOut.invalid_channels list assignment": (MaskedTimeInOut(), "obj.invalid_channels = [0]"),
#    "ChannelMixer.weights item assignment" : (ChannelMixer(weights=[[1.,2.,3.]]), "obj.weights[0] = 0."),
    "ChannelMixer.weights new array assignment" : (ChannelMixer(weights=[[1.,2.,3.]]), "obj.weights = array([0.])"),
#    "SpatialInterpolator.Q item assignment": (SpatialInterpolator(), "obj.Q[0] = 0."),
    "SpatialInterpolator.Q array assignment": (SpatialInterpolator(), "obj.Q = array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])"),
#    "SpatialInterpolatorRotation.Q item assignment": (SpatialInterpolatorRotation(), "obj.Q[0] = 0."),
    "SpatialInterpolatorRotation.Q array assignment": (SpatialInterpolatorRotation(), "obj.Q = array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])"),
#    "SpatialInterpolatorConstantRotation.Q item assignment": (SpatialInterpolatorConstantRotation(), "obj.Q[0] = 0."),
    "SpatialInterpolatorConstantRotation.Q array assignment": (SpatialInterpolatorConstantRotation(), "obj.Q = array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])"),
    "Mixer.sources item assignment": (Mixer(source=SamplesGenerator(), sources=[SamplesGenerator()]), "obj.sources[0] = SamplesGenerator(numchannels=1)"),  
    "Mixer.sources list assignment": (Mixer(sources=[SamplesGenerator()]), "obj.sources = [SamplesGenerator(), SamplesGenerator()]"),  
    "WriteWAV.channels item assignment": (WriteWAV(channels=[1]), "obj.channels[0] = 0"),  
    "WriteWAV.channels list assignment": (WriteWAV(), "obj.channels = [0]"),
    }


class Test_DigestChange(TestCase):
    """Test that ensures that digest of Acoular classes changes correctly on 
    changes of CArray and List attributes.
    """

    def get_digests(self,obj,statement):
        """A function that collects the digest of the obj before (d1) and
        after (d2) executing the statement that should yield to a change
        of the object digest.

        Parameters
        ----------
        obj : instance
            class instance that has an attribute `digest`
        statement : "str"
            a string that can be executed 

        Returns
        -------
        str, str
            digest before and after statement execution
        """
        d1 = obj.digest
        exec(statement)
        d2 = obj.digest
        return d1,d2

    def test_digest_changes_on_assignment(self):
        """ test that object digest change on statement execution """
        for test_label,(obj,statement) in UNEQUAL_DIGEST_TEST_DICT.items():
            # use subtest to get individual test result.
            # tests will not stop if one subtest fails!
            with self.subTest(test_label):
                digest_before_statement,digest_after_statement = self.get_digests(obj, statement)
                self.assertNotEqual(digest_before_statement,digest_after_statement)        



if __name__ == '__main__':
    import unittest 
    unittest.main()


