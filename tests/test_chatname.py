#!/usr/bin/env python3
"""
Chat Name Generator Tests - Real Agent
Tests using real agent with .env configuration

## How to Run
```bash
# Ensure the project root directory has a .env file containing OPENAI_API_KEY
python -m pytest tests/test_chatname.py -v -s
```
"""

import os
import sys
import uuid
from pathlib import Path

import pytest

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    # If python-dotenv is not available, try to manually load .env
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

from pantheon.memory import Memory
from pantheon.chatroom.special_agents import ChatNameGenerator


@pytest.mark.asyncio
async def test_generate_name_first_conversation():
    """Test name generation after first conversation - using real agent"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not found in environment")

    generator = ChatNameGenerator()
    memory = Memory("Test Chat")
    memory.id = str(uuid.uuid4())

    # Add real first conversation
    memory.add_messages([
        {"role": "user", "content": "I need help debugging a Python script that processes CSV files and generates reports"},
        {"role": "assistant", "content": "I'd be happy to help you debug your Python CSV processing script!"}
    ])

    # Use real agent to generate name
    result = await generator.generate_or_update_name(memory)

    # Verify result
    assert isinstance(result, str)
    assert len(result) > 3
    assert len(result) < 100
    assert memory.extra_data["name_generated"] is True
    assert memory.extra_data["last_name_generation_message_count"] == 2

    # Check if the generated name is relevant
    result_lower = result.lower()
    relevant_keywords = ["python", "csv", "debug", "script", "report", "process", "file"]
    assert any(keyword in result_lower for keyword in relevant_keywords), f"Generated name '{result}' should contain relevant keywords"

    print(f"✅ Generated name: '{result}'")


@pytest.mark.asyncio
async def test_update_name_after_threshold():
    """Test updating name after reaching threshold - using real agent"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not found in environment")

    generator = ChatNameGenerator()
    memory = Memory("Old Chat Name")
    memory.id = str(uuid.uuid4())

    # Simulate that a name was previously generated
    memory.extra_data["name_generated"] = True
    memory.extra_data["last_name_generation_message_count"] = 2

    # First add the initial 2 messages (simulating previous conversation)
    initial_messages = [
        {"role": "user", "content": "I need help with Python script"},
        {"role": "assistant", "content": "Sure, I can help!"}
    ]
    memory.add_messages(initial_messages)

    # Add 6 more new messages to reach update threshold (8 messages total)
    new_messages = [
        {"role": "user", "content": "Now I want to add machine learning features to predict data trends"},
        {"role": "assistant", "content": "Great! We can use scikit-learn to add predictive analytics"},
        {"role": "user", "content": "Which ML algorithms would work best for time series forecasting?"},
        {"role": "assistant", "content": "For time series, consider ARIMA, LSTM, or Prophet models"},
        {"role": "user", "content": "Can you help me implement a simple LSTM model?"},
        {"role": "assistant", "content": "Absolutely! Let's start with TensorFlow and Keras for the LSTM"}
    ]
    memory.add_messages(new_messages)

    # Use real agent to update name
    result = await generator.generate_or_update_name(memory)

    # Verify result
    assert isinstance(result, str)
    assert len(result) > 3
    assert len(result) < 100
    assert memory.extra_data["last_name_generation_message_count"] == 8

    # Check if the updated name reflects the new topic
    result_lower = result.lower()
    ml_keywords = ["machine", "learning", "ml", "lstm", "model", "predict", "forecast", "time", "series"]
    assert any(keyword in result_lower for keyword in ml_keywords), f"Updated name '{result}' should reflect ML topic"

    print(f"✅ Updated name: '{result}'")


@pytest.mark.asyncio
async def test_no_generation_insufficient_messages():
    """Test that name is not generated when messages are insufficient"""
    generator = ChatNameGenerator()
    memory = Memory("Original Name")
    memory.id = str(uuid.uuid4())

    # Add only one message
    memory.add_messages([
        {"role": "user", "content": "Hello"}
    ])

    # Should not generate a name
    result = await generator.generate_or_update_name(memory)

    assert result == "Original Name"
    assert "name_generated" not in memory.extra_data

    print("✅ No generation with insufficient messages")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])