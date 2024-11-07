import inspect

import acoular as ac


def get_beamformer_classes():
    beamformer_classes = []
    for _, obj in inspect.getmembers(ac):
        if inspect.isclass(obj) and issubclass(
            obj, ac.BeamformerBase) and obj != ac.BeamformerAdaptiveGrid:
            beamformer_classes.append(obj)
    return beamformer_classes
