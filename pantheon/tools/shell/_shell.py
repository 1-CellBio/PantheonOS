import asyncio
import sys
import os
import uuid


class AsyncShell:
    def __init__(
            self,
            shell: str | None = None,
            shell_args: list[str] | None = None,
            ):
        # Configure shell and command separator.
        if sys.platform.startswith("win"):
            # On Windows, disable echoing and the prompt:
            self.shell = shell or "cmd.exe"
            self.shell_args = shell_args or ["/Q", "/K", "prompt "]
            self.cmd_separator = " & "
        else:
            # On Unix-like systems, use bash and disable the prompt via PS1.
            self.shell = shell or "/bin/bash"
            self.shell_args = shell_args or []
            self.cmd_separator = "; "

        # Use binary mode everywhere.
        self.encoding = "utf-8"
        # Adjust environment if needed.
        if not sys.platform.startswith("win"):
            self.env = os.environ.copy()
            self.env["PS1"] = ""
        else:
            self.env = None
        self.process = None

    async def start(self):
        """Starts the shell process in binary mode and drains any startup output."""
        self.process = await asyncio.create_subprocess_exec(
            self.shell, *self.shell_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self.env
        )
        await self._drain_initial_output()

    async def _drain_initial_output(self):
        """
        Wait a moment for any startup output (banner/prompt) and drain it.
        """
        await asyncio.sleep(0.2)
        while True:
            try:
                line = await asyncio.wait_for(self.process.stdout.readline(), timeout=0.1)
            except asyncio.TimeoutError:
                break
            if not line:
                break
            # Decode the line for optional debugging or logging.
            line = line.decode(self.encoding)
            # Uncomment the next line to see what was drained:
            # print("Drained:", line.rstrip())

    async def run_command(self, command: str) -> str:
        """
        Sends a command to the shell (appending a unique marker) and returns all output
        up to the marker.
        """
        marker = f"__COMMAND_END_{uuid.uuid4().hex}__"
        # Append the marker with the proper command separator.
        full_command = f"{command}{self.cmd_separator}echo {marker}\n"
        self.process.stdin.write(full_command.encode(self.encoding))
        await self.process.stdin.drain()

        output_lines = []
        # Read until the marker is found.
        while True:
            line = await self.process.stdout.readline()
            if not line:
                break  # No more output (process may have ended)
            line = line.decode(self.encoding)
            if marker in line:
                break  # End of command output.
            output_lines.append(line)
        return "".join(output_lines)

    async def close(self):
        """Closes the shell process gracefully."""
        if self.process:
            exit_command = "exit\n"
            self.process.stdin.write(exit_command.encode(self.encoding))
            await self.process.stdin.drain()
            await self.process.wait()

    # Allow use as an async context manager.
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

# Example usage:
async def main():
    shell = AsyncShell()
    await shell.start()

    output = await shell.run_command("echo Hello, world!")
    print("Output of echo command:")
    print(output)

    # Use "dir" on Windows and "ls -l" on Unix-like systems.
    dir_command = "dir" if sys.platform.startswith("win") else "ls -l"
    output = await shell.run_command(dir_command)
    print("Directory listing:")
    print(output)

    # Run a wrong command.
    wrong_command = "wrong_command"
    output = await shell.run_command(wrong_command)
    print("Output of wrong command:")
    print(output)

    await shell.close()



if __name__ == '__main__':
    asyncio.run(main())
