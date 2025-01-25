from pydantic import BaseModel

from pantheon.smart_func import smart_func


async def test_smart_func():
    @smart_func(model="gpt-4o-mini")
    async def translate(text: str) -> str:
        """Translate the given text to English."""

    res = await translate("你好，世界！")
    assert res.lower() == "hello, world!"


async def test_smart_func_with_tools():
    _city = None

    def get_weather(city: str):
        nonlocal _city
        _city = city
        return {"weather": "sunny"}

    @smart_func(model="gpt-4o-mini", tools=[get_weather])
    async def get_weather_info(city: str) -> str:
        """Get the weather information of the given city."""

    await get_weather_info("Palo Alto")
    assert _city.lower() == "palo alto"


async def test_smart_func_structured_output():
    class Book(BaseModel):
        title: str
        author: str
        price: float

    @smart_func(model="gpt-4o-mini")
    async def recommend_book() -> Book:
        """Recommend a book."""

    book = await recommend_book()
    assert isinstance(book, Book)


def test_smart_func_sync():
    @smart_func(model="gpt-4o-mini")
    def translate(text: str) -> str:
        """Translate the given text to English."""

    res = translate("你好，世界！")
    assert res.lower() == "hello, world!"
