from distutils.core import setup

import acoular
bf_version = str(acoular.__version__)
bf_author = str(acoular.__author__)
    
setup(name="acoular",
      version=bf_version,
      description="Library for the acoustic beamforming",
      author=bf_author,
      author_email="ennes.sarradj@gmx.de",
      url="",
      packages = ['acoular'],
      scripts=['ResultExplorer.py','CalibHelper.py'],
      include_package_data = True,
      package_data={'acoular': ['*.pyd','*.so','xml/*.xml']}
     )
#package_data={'acoular': ['doc/*.*','*.pyd','*.so','xml/*.xml']}