from setuptools import setup, find_packages
import versioneer


setup(
    name='telluric',
    version=versioneer.get_version(),
    description='telluric API',
    author='Juan Luis Cano, Guy Doulberg, Slava Kerner, Lucio Torre, Ariel Zerahia',
    maintainer_email='juanlu@satellogic.com',
    url='https://github.com/satellogic/telluric/',
    download_url=(
        'https://github.com/satellogic/telluric/tags/' +
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
        'ipyleaflet',
        'pyproj',
        'shapely>=1.6.3,<2.*',
        'rasterio>=1.0a12',
        'scipy',
        'pillow',
        'mercantile>=0.10.0',
        'matplotlib',
    ],
    extras_require={
        ':python_version<="3.4"': ['typing'],
        'dev': [
            'coverage',
            'packaging',
            'pytest',
            'pytest-cov',
            'sphinx',
            'ipython',
            'nbsphinx',
            'sphinx_rtd_theme'
        ],
    },
    cmdclass=versioneer.get_cmdclass(),
)
