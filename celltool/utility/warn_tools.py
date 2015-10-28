# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

import warnings
import sys

_original_showwarning = warnings.showwarning

class CelltoolWarning(UserWarning):
    pass

_warning_queue = []

def _celltool_queuewarning(message, category, filename, lineno, file=sys.stderr):
    if issubclass(category, CelltoolWarning):
        _warning_queue.append('Warning: %s\n'%message)
    else:
        _original_showwarning(message, category, filename, lineno, file)

def queue_celltool_warnings():
    global _warning_queue
    _warning_queue = []
    warnings.showwarning = _celltool_queuewarning

def end_queue():
    global _warning_queue
    for warning in _warning_queue:
        sys.stderr.write(warning)
    _warning_queue = []
    warnings.showwarning = _original_showwarning

def warn(message):
    warnings.warn(message, CelltoolWarning, stacklevel = 2)
