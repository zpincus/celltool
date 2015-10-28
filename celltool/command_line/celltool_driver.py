# Copyright 2007 Zachary Pincus
# This file is part of CellTool.
# 
# CellTool is free software; you can redistribute it and/or modify
# it under the terms of version 2 of the GNU General Public License as
# published by the Free Software Foundation.

"""Run commands from the celltool suite.
"""

import optparse
import sys
import cli_tools
from celltool.utility import warn_tools
import plugins
from celltool_commands import celltool_commands

usage = "usage: %prog [options] command [arguments and options for 'command']"

parser = optparse.OptionParser(usage=usage, description=__doc__.strip(), 
    add_help_option = False, formatter=cli_tools.CelltoolFormatter())
parser.disable_interspersed_args()
parser.add_option('-d', '--debug', dest='call_handler', action='store_const',
    const=cli_tools.debug_handler, default=cli_tools.quiet_handler, 
    help='on error, print full error output')
parser.add_option('-h', '--help', action='store_true',
    help='print help text for all commands and exit')

def main(name, arguments):
    parser.prog = name
    options, args = parser.parse_args(arguments)
    if options.help:
        print_help(name, short=True)
    if len(args) == 0:
        print_help(name, short=True)
    command = args[0]
    command_name = ' '.join([name, command])
    command_args = args[1:]
    if command not in celltool_commands:
        parser.error("command '%s' is not a recognized celltool command."%command)
    full_name = 'celltool.command_line.'+command
    command_module = __import__(full_name, {}, {}, ['main'])
    options.call_handler(command_module.main, command_name, command_args)

def print_help(name, short = True):
    parser.print_help()
    print('\nValid commands are:\n')
    for command in celltool_commands:
        full_name = 'celltool.command_line.'+command
        command_module = __import__(full_name, {}, {}, ['parser'])
        command_parser = command_module.parser
        command_parser.prog = ' '.join([name, command])
        print command+":"
        print '    '+command_parser.description.split('\n')[0]
    parser.exit(2)

if __name__ == '__main__':
    import sys
    import os
    main(os.path.basename(sys.argv[0]), sys.argv[1:])