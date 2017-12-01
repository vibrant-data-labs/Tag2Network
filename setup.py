#
# python setup.py sdist
# python setup.py bdist_wheel --universal
# twine upload dist/*
# remove egg info, dist
#

from setuptools import setup
from setuptools import find_packages

setup(name='tag2network',
      version='0.0.4',
      description='Build similarity network from tagged documents',
      long_description="Build similarity network from any dataset where a set of keywords or other \
                          tags is assigned to each document. The network is build by computing \
                          cosine simiarity between each document, based on their tag sets, and \
                          thresholding the similarities to produce a network.  Includes ability to \
                          extract ngrams from blocks of text and use as tags",
      url='https://github.com/foodwebster/tag2network',
      author='foodwebster',
      author_email='foodwebster@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy', 'scipy', 'pandas', 'networkx'
      ],
      zip_safe=False)
