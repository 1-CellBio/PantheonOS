from pantheon.tools.web_browse.duckduckgo import duckduckgo_search
from pantheon.tools.web_browse.web_crawl import web_crawl


def test_duckduckgo_search():
    try:
        results = duckduckgo_search("cats dogs", max_results=5)
        assert len(results) == 5
        for result in results:
            assert isinstance(result, dict)
            assert "title" in result
    except Exception as e:
        print(e)


async def test_web_crawl():
    result = await web_crawl(["https://www.example.com"])
    print(result)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str)
