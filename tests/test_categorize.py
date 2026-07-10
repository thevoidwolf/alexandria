from alexandria.categorize import categorize_url
from alexandria.config import CategoryRule, Config


def _cfg_with(rules: tuple[CategoryRule, ...]) -> Config:
    return Config(home=None, category_rules=rules)  # type: ignore[arg-type]


def test_user_supplied_category_wins():
    cfg = _cfg_with((CategoryRule("arxiv.org", "research", ("paper",)),))
    cat, tags = categorize_url(
        "https://arxiv.org/abs/1234", cfg,
        supplied_category="notes", supplied_tags=["mine"],
    )
    assert cat == "notes"
    assert tags == ["mine"]


def test_exact_host_match_applies_rule():
    cfg = _cfg_with((CategoryRule("arxiv.org", "research", ("paper",)),))
    cat, tags = categorize_url("https://arxiv.org/abs/1234", cfg)
    assert cat == "research"
    assert tags == ["paper"]


def test_wildcard_host_match():
    cfg = _cfg_with((CategoryRule("*.pge.com", "bills", ("utility", "electric")),))
    cat, tags = categorize_url("https://my.account.pge.com/bill", cfg)
    assert cat == "bills"
    assert tags == ["utility", "electric"]


def test_first_matching_rule_wins():
    cfg = _cfg_with((
        CategoryRule("arxiv.org", "research", ("paper",)),
        CategoryRule("*", "everything", ("catchall",)),
    ))
    cat, _ = categorize_url("https://arxiv.org/abs/x", cfg)
    assert cat == "research"


def test_supplied_tags_merge_with_rule_tags():
    cfg = _cfg_with((CategoryRule("arxiv.org", "research", ("paper",)),))
    cat, tags = categorize_url(
        "https://arxiv.org/abs/x", cfg,
        supplied_tags=["favourite"],
    )
    assert cat == "research"
    assert tags == ["favourite", "paper"]


def test_no_rule_match_returns_none_category():
    cfg = _cfg_with((CategoryRule("arxiv.org", "research", ()),))
    cat, tags = categorize_url("https://example.com/x", cfg)
    assert cat is None
    assert tags == []
