from setuptools import setup
from setuptools import find_packages

# Sync the env.yml file here

setup(
    name="Modti",
    version="0.0.1",
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    # entry_points={
    #     "console_scripts": [
    #         "thera=thera.apps.cli:cli",
    #     ]
    # },
)
