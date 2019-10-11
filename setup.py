from setuptools import setup, find_packages


def load_readme():
    with open('README.md') as file:
        return file.read()


setup(
    name='touvlo',
    version='0.4.3',
    setup_requires=['pbr'],
    keywords='Machine Learning ML algorithms block scratch',
    install_requires=['wheel', 'cmake', 'pbr', 'numpy'],
    pbr=True,
    zip_safe=False,
    packages=find_packages(exclude=("tests",)),
    test_suite="nose.collector",
    tests_require=["nose"]
)
