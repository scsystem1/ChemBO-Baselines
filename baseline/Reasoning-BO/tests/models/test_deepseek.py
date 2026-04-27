from src.llms.deepseek import DeepSeekClient


def test_generate():
    """Test single-round generation"""
    dsclient = DeepSeekClient()
    content, reasoning_content = dsclient.generate(
        "What's the highest mountain in the world?", max_tokens=100
    )
    assert content
    assert reasoning_content


# def test_multi_round_generate():
#     """Test multi-round generation"""
#     dsclient = DeepSeekClient()
#     content, reasoning_content = dsclient.multi_round_generate(
#         "What's the highest mountain in the world?", max_tokens=100
#     )
#     assert content
#     assert reasoning_content


# def test_multi_round_stream_generate():
#     """Test multi-round stream generation"""
#     dsclient = DeepSeekClient()
#     content, reasoning_content = dsclient.multi_round_stream_generate(
#         "What's the highest mountain in the world?", max_tokens=100
#     )
#     assert content
#     assert reasoning_content
