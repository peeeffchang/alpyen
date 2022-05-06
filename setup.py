#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0',
    'ib-insync',
    'eventkit',
    'pandas',
    'asyncio',
    'arch',
    'numpy',
    'gemini_python'
]

test_requirements = ['pytest>=3', ]

setup(
    author="P.F. Chang",
    author_email='peeeffchang@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Python based algo trading platform for IB.",
    entry_points={
        'console_scripts': [
            'alpyen=alpyen.cli:main',
        ],
    },
    project_urls={
        'Documentation': 'https://github.com/peeeffchang/alpyen/blob/main/README.rst',
        'Source': 'https://github.com/peeeffchang/alpyen/',
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description_content_type="text/markdown",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='alpyen',
    name='alpyen',
    packages=find_packages(include=['alpyen', 'alpyen.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/peeeffchang/alpyen',
    version='1.2.0',
    zip_safe=False,
)
