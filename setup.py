from setuptools import setup, find_packages

setup(
    name='Raven Distribution Framework',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'python-socketio==4.5.1',
        'requests==2.23.0',
        'SQLAlchemy==1.3.17',
        'redis==3.5.3'
    ],
)
