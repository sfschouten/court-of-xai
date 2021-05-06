import distutils
import io
import os
import subprocess

from setuptools import setup

class PylintCommand(distutils.cmd.Command):
  '''A custom command to run Pylint to check for errors on all source files'''

  # Required to implement 
  user_options = []

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    command = ['pylint']
    command.append(f'--rcfile={os.path.abspath(".pylintrc")}')
    command.append('xai_court')
    self.announce(f'Running command: {command}', level=distutils.log.INFO)
    subprocess.check_call(command)


# Package meta-data
NAME = 'court-of-xai'
DESCRIPTION = 'A framework to design and run experiments to calculate different correlation metrics between feature additive explanation methods'
URL = 'https://github.com/sfschouten/court-of-xai'
EMAIL = 'michael@mneely.tech, sfschouten@gmail.com'
AUTHOR = 'Michael J. Neely, Stefan F. Schouten'
REQUIRES_PYTHON = '>=3.6'
VERSION = '1.0.0'

REQUIRED = [
  'pandas',
  'nltk',
  'tqdm',
  'numpy',
  'allennlp==1.1.0',
  'allennlp-models>=1.0.0',
  'scipy',
  'seaborn',
  'gensim',
  'spacy==2.1.9',
  'matplotlib',
  'ipython',
  'scikit_learn',
  'entmax',
  'requests',
  'python-json-logger',
  'captum>=0.3.1',
  'pyircor',
  'llvmlite==0.32.1'
]

SETUP = [
  'pytest-runner',
  'torch',
  'torchtext'
]

TEST = [
  'pylint',
  'pytest',
  'pytest-env',
  'pytest-pylint'
]

EXTRAS = {
  'testing': TEST
}

HERE = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description
try:
  with io.open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = '\n' + f.read()
except FileNotFoundError:
  LONG_DESCRIPTION = DESCRIPTION

setup(
  name=NAME,
  version=VERSION,
  description=DESCRIPTION,
  long_description=LONG_DESCRIPTION,
  long_description_content_type='text/markdown',
  author=AUTHOR,
  author_email=EMAIL,
  python_requires=REQUIRES_PYTHON,
  url=URL,
  py_modules=['xai_court'],
  install_requires=REQUIRED,
  setup_requires=SETUP,
  tests_require=TEST,
  extras_require=EXTRAS,
  include_package_data=True,
  license='MIT',
  cmdclass={
    'lint': PylintCommand
  },
  classifiers=[
    # Trove classifiers
    # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    'License :: Other/Proprietary License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: Implementation :: CPython',
    'Programming Language :: Python :: Implementation :: PyPy'
  ]
)
