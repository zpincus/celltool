from celltool.utility import path
import os, sys

def find_plugins(directory):
  if not directory.exists():
    return []
  plugin_files = [p.namebase for p in directory.files('*.py')]
  plugins = [p for p in plugin_files if p not in ('__init__', 'setup')]
  plugins += [d.name for d in directory.dirs() if (d / '__init__.py').exists()]
  return plugins
  
def _monkeypatch_command(plugin_mod, plugin_name):
  import celltool.command_line.celltool_commands
  celltool.command_line.celltool_commands.celltool_commands.append(plugin_name)
  setattr(celltool.command_line, plugin_name, plugin_mod)
  sys.modules['celltool.command_line.'+plugin_name] = plugin_mod

py2exe_plugins = None
if  hasattr(sys,"frozen") and sys.frozen == 'console_exe':
  py2exe_plugins = path.path(sys.executable).parent.abspath() / 'Plugins'
  _plugins = find_plugins(py2exe_plugins)
  sys.path.append(py2exe_plugins)
else:
  _plugins = find_plugins(path.path(__file__).parent)

celltool_plugins = None
try:
  celltool_plugins = os.environ['CELLTOOL_PLUGINS']
  _plugins += find_plugins(path.path(celltool_plugins))
  sys.path.append(celltool_plugins)
except:
  pass

for plugin in _plugins:
  plugin_mod = __import__(plugin, globals(), locals())
  if hasattr(plugin_mod, '_COMMAND_PLUGIN'):
      _monkeypatch_command(plugin_mod, plugin)

if celltool_plugins is not None:
  sys.path.remove(celltool_plugins)
if py2exe_plugins is not None:
  sys.path.remove(py2exe_plugins)  
del find_plugins, celltool_plugins, py2exe_plugins, path, os, sys
try:
  del plugin
except:
  pass