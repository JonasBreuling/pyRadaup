# from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

# setup(
#     name="pyRadaup",
#     version="0.0.1",
#     author="Jonas Breuling",
#     author_email="jonas.breuling@inm.uni-stuttgart.de",
#     description="Python interface to Radaup high-performance stiff ODE / DAE solver",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/TODO",
#     packages=find_packages(),
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
#         "Operating System :: OS Independent",
#     ],
#     python_requires=">=3.6",
#     install_requires=["numpy", "scipy"],
# )

# # run "python setup.py develop" once !

from numpy.distutils.core import Extension

# ext1 = Extension(name = 'scalar',
#                  sources = ['scalar.f'])
ext_radau = Extension(
    name="radau",
    sources=[
        "pyRadaup/fortran/radau.pyf",
        "pyRadaup/fortran/radau.f",
        "pyRadaup/fortran/dc_decsol.f",
        "pyRadaup/fortran/decsol.f",
    ],
)

if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(
        name="radau",
        description="F2PY Users Guide examples",
        author="Pearu Peterson",
        author_email="pearu@cens.ioc.ee",
        ext_modules=[ext_radau],
    )

# python setup.py build
# python -m pip install .
