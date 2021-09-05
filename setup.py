from setuptools import setup, find_packages

setup(
    name='recotut',
    version='0.0.1',
    description='RecSys package',
    author='Sparsh Agarwal',
    author_email='recohut@gmail.com',
    url='https://github.com/sparsh-ai',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'PyYAML',
        'pandas>=0.23.3',
        'numpy>=1.14.5'
    ],
    extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Education',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Topic :: Recommendation Systems',
      ]
)