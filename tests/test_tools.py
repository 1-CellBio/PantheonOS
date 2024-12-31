from synago.tools.duckduckgo import duckduckgo_search
from synago.tools.web_crawl import web_crawl


def test_duckduckgo_search():
    results = duckduckgo_search("cats dogs", max_results=5)
    assert len(results) == 5
    for result in results:
        assert isinstance(result, dict)
        assert "title" in result


async def test_web_crawl():
    result = await web_crawl(["https://www.example.com"])
    print(result)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)
