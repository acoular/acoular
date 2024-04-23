# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements testing of frequency beamformers.
"""

import unittest
import acoular as ac
from traits.api import HasTraits, Int, Float, Bool, Range, TraitEnum, Enum

class Test_Instancing(unittest.TestCase):
    """Test that ensures that digest of Acoular classes changes correctly on
    changes of CArray and List attributes.
    """

    def test_instancing(self):
        """ test that all Acoular classes can be instatiated """
        # iterate over all Acoular definitions labels
        for i in dir(ac):
            with self.subTest(i):
                j = getattr(ac,i) # class, function or variable
                if isinstance(j,type): # is this a class ?
                    j() # this is an instance of the class

    def test_set_traits(self):
        """ test that important traits can be set"""
        # iterate over all Acoular definitions labels
        for i in dir(ac):
            j = getattr(ac,i) # class, function or variable
            # HasTraits derived class ?
            if isinstance(j,type) \
                and issubclass(j,HasTraits) \
                and ('digest' in j.class_traits().keys()):
                do = j.class_traits()['digest'].depends_on
                if do:
                    obj = j()
                    for k in do:
                        with self.subTest(i+'.'+k):
                            if k in j.class_trait_names():
                                tr = j.class_traits()[k]
                                # handling different Trait types
                                # todo: use hypothesis based setattr
                                if tr.is_trait_type(Int):
                                    setattr(obj,k,1)
                                elif tr.is_trait_type(Float):
                                    setattr(obj,k,0.1)
                                elif tr.is_trait_type(Bool):
                                    setattr(obj,k,False)
                                elif tr.is_trait_type(Range):
                                    low = tr.handler._low
                                    high = tr.handler._high
                                    setattr(obj,k,(high+low)/2)
                                elif tr.is_trait_type(TraitEnum) or tr.is_trait_type(Enum):
                                    v = tr.handler.values
                                    setattr(obj,k,v[len(v)//2])

if __name__ == '__main__':
    unittest.main()
