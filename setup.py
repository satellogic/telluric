from setuptools import setup, find_packages
import versioneer


setup(
    name='telluric',
    version=versioneer.get_version(),
    description='Interactive geospatial data manipulation in Python',
    license="MIT",
    long_description=open('README.md', encoding='utf-8').read(),
    author='Juan Luis Cano, Guy Doulberg, Slava Kerner, Lucio Torre, Ariel Zerahia, Denis Rykov',
    maintainer_email='juanlu@satellogic.com',
    url='https://github.com/satellogic/telluric/',
    download_url=(
        'https://github.com/satellogic/telluric/tags/v' +
        versioneer.get_version()
    ),
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.4",
    install_requires=[
        'affine',
        'dsnparse',
        'fiona>=1.7.2,<2.*',
        'folium',
        'ipyleaflet!=0.8.2',
        'pyproj',
        'shapely>=1.6.3,<2.*',
        'rasterio>=1.0.3.post1,<2.*',
        'pillow',
        'mercantile>=0.10.0',
        'matplotlib',
        'boltons',
        'imageio',
    ],
    extras_require={
        ':python_version<="3.4"': ['typing', 'pandas<0.21.0'],
        'dev': [
            'coverage',
            'mypy',
            'packaging',
            'pycodestyle',
            'pytest',
            'pytest-cov',
            'sphinx',
            'ipython',
            'nbsphinx',
            'sphinx_rtd_theme'
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    zip_safe=False,
    long_description_content_type='text/markdown',
    cmdclass=versioneer.get_cmdclass(),
)
