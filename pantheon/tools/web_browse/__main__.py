from . import WebBrowseToolSet
import asyncio

toolset = WebBrowseToolSet("web_browse")
asyncio.run(toolset.run())
