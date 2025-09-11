from pathlib import Path
from setuptools import setup, find_packages

readme = Path(__file__).with_name('README.md').read_text(encoding='utf-8')

setup(
  name='torchspd',
  version='0.1.0',
  description='SPD matrix functions for PyTorch',
  long_description=readme,
  long_description_content_type='text/markdown',
  author='Samuel Boïté',
  license='MIT',
  url='https://github.com/samuelbx/torchspd',
  project_urls={
    'Source': 'https://github.com/samuelbx/torchspd',
    'Tracker': 'https://github.com/samuelbx/torchspd/issues',
  },
  python_requires='>=3.9',
  install_requires=['torch>=2.1'],
  extras_require={
    'test': ['pytest>=7', 'scipy>=1.10; python_version<"3.13"'],
    'dev': ['pytest>=7', 'ruff>=0.4', 'mypy>=1.7'],
  },
  packages=find_packages(where='src', include=['torchspd', 'torchspd.*']),
  package_dir={'': 'src'},
  include_package_data=True,
  license_files=['LICENSE'],
  keywords=['pytorch', 'spd', 'matrix functions', 'linear algebra', 'logm', 'sqrtm'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Operating System :: OS Independent',
  ],
  zip_safe=False,
)
