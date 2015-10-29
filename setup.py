import distutils.core
packages = '''
celltool
celltool.command_line
celltool.command_line.plugins
celltool.contour
celltool.numerics
celltool.plot
celltool.utility
'''.strip().split()

distutils.core.setup(name='celltool', version='2.0', description='celltool package',
    packages=packages, scripts=['celltool/command_line/celltool'])
