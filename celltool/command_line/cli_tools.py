# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

import traceback
import sys
import exceptions
from celltool.utility import warn_tools
from celltool.utility import optparse
import numpy
import glob
import os

def debug_handler(function, *args, **kws):
    warn_tools.queue_celltool_warnings()
    numpy.seterr(all='raise')
    try:
        ret = function(*args, **kws)
        warn_tools.end_queue()
        return ret
    except Exception, e:
        if isinstance(e, exceptions.SystemExit):
            raise e
        warn_tools.end_queue()
        traceback.print_exc()
        sys.exit(1)

def quiet_handler(function, *args, **kws):
    warn_tools.queue_celltool_warnings()
    try:
        ret = function(*args, **kws)
        warn_tools.end_queue()
        return ret
    except Exception, e:
        warn_tools.end_queue()
        if isinstance(e, exceptions.SystemExit):
            raise e
        type, value, tb = sys.exc_info()
        print ''.join(traceback.format_exception_only(type, value))
        sys.exit(1)

class CelltoolFormatter(optparse.TitledHelpFormatter):
    def __init__(self, indent_increment=0, max_help_position=24, width=None, short_first=True):
        optparse.TitledHelpFormatter.__init__(self, indent_increment, max_help_position, width, short_first)
    def format_description(self, description):
        if description:
            return description + "\n"
        else:
            return ""


def glob_args(args):
    outargs = []
    for arg in args:
        outargs.extend(my_glob(arg))
    return outargs


def my_glob(pathname):
    """Like glob.glob, but doesn't filter out invalid file names.
    We want to catch that error explicitly to avoid user confusion."""
    if not glob.has_magic(pathname):
        yield pathname
        return
    dirname, basename = os.path.split(pathname)
    if not dirname:
        for name in glob.glob1(os.curdir, basename):
            yield name
        return
    if glob.has_magic(dirname):
        dirs = my_glob(dirname)
    else:
        dirs = [dirname]
    if glob.has_magic(basename):
        glob_in_dir = glob.glob1
    else:
        glob_in_dir = glob.glob0
    for dirname in dirs:
        for name in glob_in_dir(dirname, basename):
            yield os.path.join(dirname, name)
