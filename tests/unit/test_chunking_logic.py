from summarizer import RepositorySummarizer
from utils import chunk_text_by_lines


class DummyGitHubService:
    pass


class CountingLLMService:
    def __init__(self):
        self.calls = 0

    def call_json(self, **_kwargs):
        self.calls += 1
        return {"chunk_summary": [f"chunk-{self.calls}"]}


def test_chunking_count_and_merge_calls():
    llm = CountingLLMService()
    summarizer = RepositorySummarizer(github_service=DummyGitHubService(), llm_service=llm)

    # Large enough body to force multiple chunks with default token approximation.
    content = "\n".join(f"line {i} abcdefghijklmnopqrstuvwxyz" for i in range(4000))
    chunks = chunk_text_by_lines(content)
    assert len(chunks) > 1

    merged = summarizer._summarize_large_file("src/huge.py", content)

    assert llm.calls == len(chunks)
    assert merged.count("- chunk-") == len(chunks)
