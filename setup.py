from distutils.core import setup
    
setup(name="beamfpy",
      version="3.0alpha",
      description="Library for the acoustic beamforming",
      author="Ennes Sarradj",
      author_email="ennes.sarradj@gmx.de",
      url="",
      packages = ['beamfpy','beamfpy.scripts'],
      scripts=['ResultExplorer.py','CalibHelper.py'],
      include_package_data = True,      
      package_data={'beamfpy': ['doc/*.*','*.pyd','*.so','xml/*.xml']}
     )
