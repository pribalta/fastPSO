import setuptools

setuptools.setup(
    name='fastPSO',
    version='0.0.2',
    author='Pablo Ribalta Lorenzo',
    author_email='pribalta@ieee.org',
    packages=['fastPSO', 'test'],
    url='https://github.com/pribalta/fastPSO',
    license='LICENSE.txt',
    description='Fast parallel PSO library for Python with support for CPU and GPU multithreading.',
    long_description=open('README.md').read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
