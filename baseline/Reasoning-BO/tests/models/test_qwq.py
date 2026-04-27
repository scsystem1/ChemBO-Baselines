from src.llms.qwq import QWQClient


def test_generate():
    """Test single-round generation"""
    qwqclient = QWQClient()
    content, reasoning_content = qwqclient.generate(
        "What's the highest mountain in the world?", max_tokens=100
    )
    assert content
    assert reasoning_content


def test_multi_round_generate():
    """Test multi-round generation"""
    qwqclient = QWQClient()
    # First round
    content1, reasoning_content1 = qwqclient.multi_round_generate(
        "What's the highest mountain in the world?", max_tokens=100
    )
    assert content1
    assert reasoning_content1
    # Second round
    content2, reasoning_content2 = qwqclient.multi_round_generate(
        "How tall is it?", max_tokens=100
    )
    assert content2
    assert reasoning_content2


def test_view_messages():
    """Test viewing message history"""
    qwqclient = QWQClient()
    qwqclient.generate("Test message", max_tokens=10)
    # Just verify the method runs without error
    qwqclient.view_messages()


def test_save_messages(tmp_path):
    """Test saving messages to file"""
    import os

    qwqclient = QWQClient()
    qwqclient.generate("Test save message", max_tokens=10)
    test_file = tmp_path / "test_messages.jsonl"
    qwqclient.save_messages(str(test_file))
    assert os.path.exists(test_file)
    assert os.path.getsize(test_file) > 0
