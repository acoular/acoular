from distutils.core import setup
#~ from distutils.command.install import INSTALL_SCHEMES

#~ for scheme in INSTALL_SCHEMES.values():
    #~ scheme['data'] = scheme['purelib'] 
    
setup(name="beamfpy",
      version="2.0alpha",
      description="Library for the acoustic beamforming",
      author="Ennes Sarradj",
      author_email="ennes.sarradj@gmx.de",
      url="",
      packages = ['beamfpy','beamfpy.scripts'],
      scripts=['ResultExplorer.py'],
      include_package_data = True,      
      package_data={'beamfpy': ['doc/*.*','*.pyd','xml/*.xml']}
#                    ('beamfpy',['./beamfpy/beamformer.pyd'])]
     )
