from crawl4ai import AsyncWebCrawler
import asyncio


async def web_crawl(urls: list[str]) -> list[str]:
    """
    Crawl the web and return the contents of the pages.
    Result will be in markdown format.
    """
    async with AsyncWebCrawler(verbose=True) as crawler:
        tasks = [crawler.arun(url=url) for url in urls]
        results = await asyncio.gather(*tasks)
    contents = []
    for result in results:
        try:
            contents.append(result.markdown_v2.raw_markdown)
        except Exception as e:
            print(e)
            contents.append("")
    return contents
