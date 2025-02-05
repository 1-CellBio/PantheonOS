import uuid

from ._shell import AsyncShell
from ...remote import ToolSet, tool


class ShellToolSet(ToolSet):
    def __init__(
            self,
            name: str,
            worker_params: dict | None = None,
            ):
        super().__init__(name, worker_params)
        self.clientid_to_shellid = {}
        self.shells = {}

    @tool
    async def new_shell(self) -> str:
        """Create a new shell and return its id.
        You can use `run_command_in_shell` to run command in the shell,
        by providing the shell id. """
        shell = AsyncShell()
        await shell.start()
        shell_id = str(uuid.uuid4())
        self.shells[shell_id] = shell
        return shell_id

    @tool
    async def close_shell(self, shell_id: str):
        """Close a shell.

        Args:
            shell_id: The id of the shell to close.
        """
        shell = self.shells[shell_id]
        await shell.close()
        del self.shells[shell_id]

    @tool
    async def run_command_in_shell(self, command: str, shell_id: str):
        """Run a command in a shell.

        Args:
            command: The command to run.
            shell_id: The id of the shell to run the command in.

        Returns:
            The output of the command.
        """
        shell = self.shells[shell_id]
        output = await shell.run_command(command)
        return output

    @tool
    async def run_command(self, command: str, __client_id__: str | None = None):
        """Run shell command and get the output.

        Args:
            command: The command to run.
        """
        if __client_id__ is not None:
            shell_id = self.clientid_to_shellid.get(__client_id__)
            if (shell_id is None) or (shell_id not in self.shells):
                shell_id = await self.new_shell()
                self.clientid_to_shellid[__client_id__] = shell_id
        else:
            shell_id = await self.new_shell()
        return await self.run_command_in_shell(command, shell_id)

