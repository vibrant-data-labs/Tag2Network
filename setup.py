# python setup.py sdist
# python setup.py bdist_wheel --universal
# twine upload dist/*

from setuptools import setup
from setuptools import find_packages

setup(name='tag2network',
      version='0.0.3',
      description='Build similarity network from tagged documents',
      url='https://github.com/foodwebster/tag2network',
      author='foodwebster',
      author_email='foodwebster@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy', 'scipy', 'pandas', 'networkx'
      ],
      zip_safe=False)
