from setuptools import setup, find_packages
from os import walk, path

docfiles = walk('doc')

setup(name="beamfpy",
      version="1.0beta",
      description="Library for the acoustic beamforming",
      author="Ennes Sarradj",
      author_email="ennes.sarradj@gmx.de",
      url="",
      packages = find_packages('.'),
#      packages_dir = {'': 'beamfpy'},
 #     py_modules=['beamfpy','nidaqimport','mplwidget'],
      scripts=['ResultExplorer.py'],
      data_files=[#('doc', ),
                    ('beamfpy',['beamformer.pyd'])]
     )
