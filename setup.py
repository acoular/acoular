from distutils.core import setup

import beamfpy
bf_version = str(beamfpy.__version__)
bf_author = str(beamfpy.__author__)
    
setup(name="beamfpy",
      version=bf_version,
      description="Library for the acoustic beamforming",
      author=bf_author,
      author_email="ennes.sarradj@gmx.de",
      url="",
      packages = ['beamfpy','beamfpy.scripts'],
      scripts=['ResultExplorer.py','CalibHelper.py'],
      include_package_data = True,
      package_data={'beamfpy': ['*.pyd','*.so','xml/*.xml']}
     )
#package_data={'beamfpy': ['doc/*.*','*.pyd','*.so','xml/*.xml']}