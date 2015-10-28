# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('command_line',parent_package,top_path)
    config.add_scripts(['celltool'])
    config.add_subpackage('plugins')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
