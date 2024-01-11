from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        print("TensorArray developed.")
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        print("TensorArray installed.")
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION

setup(
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
