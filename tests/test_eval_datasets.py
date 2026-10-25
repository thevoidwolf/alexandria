"""SciFact loader: rows from the BEIR parquet/TSV files become a Dataset."""
from evals.datasets import scifact

CORPUS = [
    {"_id": "1", "title": "Title one", "text": "Body one."},
    {"_id": "2", "title": "", "text": "Body two."},
]
QUERIES = [
    {"_id": "10", "title": "", "text": "claim ten"},
    {"_id": "11", "title": "", "text": "a train-split claim"},
]
QRELS = "query-id\tcorpus-id\tscore\n10\t1\t1\n10\t2\t1\n\n"


def test_body_is_title_blank_line_text():
    ds = scifact.build(CORPUS, QUERIES, QRELS)
    assert ds.docs["1"] == "Title one\n\nBody one."


def test_untitled_doc_is_just_its_text():
    ds = scifact.build(CORPUS, QUERIES, QRELS)
    assert ds.docs["2"] == "Body two."


def test_only_queries_with_qrels_are_kept():
    ds = scifact.build(CORPUS, QUERIES, QRELS)
    assert ds.queries == {"10": "claim ten"}


def test_qrels_parse_without_the_header_or_blank_lines():
    ds = scifact.build(CORPUS, QUERIES, QRELS)
    assert ds.qrels == {"10": {"1": 1, "2": 1}}
    assert ds.name == "scifact"
