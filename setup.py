from setuptools import setup, find_packages

setup(name='AlphaZeroDavis',
      version='v0.2',
      description='Alpha Zero implementation for various games',
      url='https://github.com/LorenzoM1997/AlphaZero',
      author='Lorenzo Mambretti',
      author_email='lmambretti@ucdavis.edu',
      license='MIT License',
      packages=find_packages(),
      zip_safe=False,
      install_requires=['tensorflow'],
      dependency_links=[]
)
