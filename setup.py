from setuptools import setup, find_packages

REQUIRED = ['gym>=0.25', 'numpy>=1.19.2', 'openpyxl>=3.0.10', 'pettingzoo>=1.20', 'pygame>=2.1.2']

setup(
    name='stunning_robots',
    version='1.0.0',
    packages=[package for package in find_packages() if package.startswith("stunning_robots")],
    install_requires=REQUIRED,
    author='Anoxiacxy',
    author_email='anoxiacxy@gmail.com',
    url='https://github.com/Anoxiacxy/Stunning-Robots',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    description='MARL environments written with pettingzoo.'
)
