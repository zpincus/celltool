def configuration(parent_package='io',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('matlab_io', parent_package, top_path)
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
