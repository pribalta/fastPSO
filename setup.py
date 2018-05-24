from distutils.core import setup

setup(
    name='pyPso',
    version='0.9',
    author='Pablo Ribalta Lorenzo',
    author_email='pribalta@ieee.org',
    packages=['pyPso', 'test'],
    url='https://github.com/pribalta/pyPSO',
    license='LICENSE.txt',
    description='Fast parallel PSO library for Python with support for CPU and GPU multithreading.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.12.1"
    ],
)
