from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt") as f:
        return f.readlines()


setup(
    name="realdeblur",
    version="0.0.1",
    url="https://github.com/janglucky/RealDeblur.git",
    description="No-text PASD baseline for paired image deblurring.",
    packages=find_packages(),
    install_requires=read_requirements(),
)
