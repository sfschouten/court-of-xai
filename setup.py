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
    command.append('ane_research')
    self.announce(f'Running command: {command}', level=distutils.log.INFO)
    subprocess.check_call(command)


# Package meta-data
NAME = 'ane-research-fact-ai'
DESCRIPTION = 'UVA FACT-AI Course Project - Attention is Not Explanation'
URL = 'https://github.com/michaeljneely/fact-ai-2020'
EMAIL = 'michael.neely@student.uva.nl'
AUTHOR = 'Michael Neely, Stefan F. Schouten, Nikolas Erkinheimo'
REQUIRES_PYTHON = '~=3.7.0'
VERSION = '0.0.1'

REQUIRED = [
  'pandas',
  'nltk',
  'tqdm',
  'numpy',
  'allennlp',
  'scipy',
  'seaborn',
  'gensim',
  'spacy',
  'matplotlib',
  'ipython',
  'scikit_learn',
  'entmax',
  'shap',
]

SETUP = [
  'pytest-runner'
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
  py_modules=['ane_research'],
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
