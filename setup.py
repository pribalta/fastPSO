from distutils.core import setup

setup(
    name='pyPso',
    version='0.9',
    author='Pablo Ribalta Lorenzo',
    author_email='pribalta@ieee.org',
    packages=['pyPso', 'pyPso.test'],
    url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Fast PSO library for Python with support for CPU and GPU multithreading.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.12.1"
    ],
)