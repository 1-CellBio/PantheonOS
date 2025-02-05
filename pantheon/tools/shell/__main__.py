from .shell import ShellToolSet
from ...remote import toolset_cli


toolset_cli(ShellToolSet, "shell")
