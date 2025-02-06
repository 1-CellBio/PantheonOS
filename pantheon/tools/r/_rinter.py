import asyncio
from ..shell._shell import AsyncCommandLineInterpreter


class AsyncRInterpreter(AsyncCommandLineInterpreter):
    def __init__(
            self,
            r_executable: str = "R",
            r_args: list[str] | None = None,
            marker: str = "__COMMAND_FINISHED__",
            ):
        super().__init__(
            executable=r_executable,
            args=r_args or ["--no-save", "--no-restore"],
            marker=marker,
        )

    async def run_command(self, command, timeout=10):
        """
        Send a command to the R interpreter and wait until the unique marker is detected.

        The marker is injected via a `cat()` command and then filtered out from the output.

        Parameters:
            command (str): The R command to execute.
            timeout (int, optional): Maximum time in seconds to wait for the marker.

        Returns:
            str: The combined output of the command (excluding the injected marker).
        """
        if self.process.returncode is not None:
            raise Exception("R interpreter process has terminated.")
        full_command = f"tryCatch({{{command}}}, error = function(e) {{message('Error: ', e$message) }})\ncat('{self.marker}\\n')\n"
        self.process.stdin.write(full_command.encode('utf-8'))
        await self.process.stdin.drain()
        return await self.read_until_marker(timeout=timeout)

    async def close(self):
        """
        Gracefully close the R interpreter.
        """
        if self.process.returncode is None:
            # Send R's quit command.
            self.process.stdin.write(b"q()\n")
            await self.process.stdin.drain()
            # R may prompt "Save workspace image? [y/n/c]:", so we send "n".
            self.process.stdin.write(b"n\n")
            await self.process.stdin.drain()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()

    def stop_on_line(self, line: str, marker: str):
        if (marker in line) and (not line.startswith("> ")):
            return True
        return False

    def filter_out_line(self, line: str, marker: str):
        if line.startswith("> ") or line.startswith("+ "):
            return True
        return False

# Example usage demonstrating both send_command and directly calling read_until_marker.
async def main():
    # On Windows, adjust the executable name as needed (e.g., "R.exe" or "Rterm.exe").
    r = AsyncRInterpreter(r_executable="R.exe")
    initial_output = await r.start()
    print("Initial output:")
    print(initial_output)
    try:
        # Send a command that prints a message.
        output, _ = await r.run_command("print('Hello from R!')", timeout=10)

        print("Output from run_command:")
        print(output)

        # Send another command.
        output, _ = await r.run_command("sum(1:10)", timeout=10)
        print("Output from run_command:")
        print(output)

        # send a command with error
        output, _ = await r.run_command("sum(1:10) + a", timeout=10)
        print("Output from run_command:")
        print(output)

        # send a variable
        output, _ = await r.run_command("a <- 1", timeout=10)
        print("Output from run_command:")
        print(output)

        # send a variable
        output, _ = await r.run_command("a", timeout=10)
        print("Output from run_command:")
        print(output)

        # send a multi-line command
        output, _ = await r.run_command("""
        print('Hello from R!')
        print('Hello from R!')
        print('Hello from R!')
        """, timeout=10)
        print("Output from run_command:")
        print(output)

        # Alternatively, if you need to read the stdout directly (for commands that do not automatically inject the marker),
        # you can call read_until_marker after manually sending a marker.
        #
        # For demonstration, we send a print command and then inject the marker.
        r.process.stdin.write(b"print('Another message\\n')\n")
        r.process.stdin.write(b"cat('__COMMAND_FINISHED__\\n')\n")
        await r.process.stdin.drain()
        direct_output, _ = await r.read_until_marker(timeout=10)
        print("Output from direct read_until_marker call:")
        print(direct_output)
    finally:
        await r.close()


if __name__ == "__main__":
    asyncio.run(main())
