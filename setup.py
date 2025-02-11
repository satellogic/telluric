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
    python_requires=">=3.9",
    install_requires=[
        'affine',
        'fiona>=1.8.4,<1.9.0',
        'shapely>=1.6.3,<2.0.0',
        'rasterio>=1.2.0,<1.4.0',
        'pillow',
        'mercantile>=0.10.0',
        'boltons',
        'imageio',
        "lxml",
        'python-dateutil',
        'numpy<2.0',
    ],
    extras_require={
        'dev': [
            'coverage',
            'mypy',
            'types-python-dateutil',
            'packaging',
            'pycodestyle',
            'pytest>=4',
            'pytest-cov<=2.8.1',
            'sphinx',
            'ipython',
            'nbsphinx',
            'sphinx_rtd_theme'
        ],
        'vis': [
            'ipyleaflet!=0.8.2',
            'matplotlib',
            'folium>=0.6.0',
            'tornado',
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    zip_safe=False,
    long_description_content_type='text/markdown',
    cmdclass=versioneer.get_cmdclass(),
)
