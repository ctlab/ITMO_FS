from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name='ITMO_FS',
    version='0.1',
    packages=find_packages(),
    description='Python Feature Selection library from ITMO University',
    long_description_content_type="text/markdown",
    long_description=long_description,

    author="Nikita Pilnenskiy",
    author_email="somacruz@bk.ru",
    install_requires=['numpy', 'utils', 'scipy']
)
