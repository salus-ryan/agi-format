"""Tests for the formal verification module."""

from runtime.verification import FormalVerifier


class TestFormalVerifier:
    def test_valid_python(self):
        fv = FormalVerifier()
        results = fv.verify_code("def foo(x):\n    return x + 1")
        assert len(results) >= 1
        assert all(r.verified for r in results)

    def test_syntax_error(self):
        fv = FormalVerifier()
        results = fv.verify_code("def foo(x:\n    return x")
        syntax_results = [r for r in results if r.backend == "ast_parse"]
        assert len(syntax_results) == 1
        assert not syntax_results[0].verified

    def test_empty_code(self):
        fv = FormalVerifier()
        results = fv.verify_code("")
        assert isinstance(results, list)

    def test_summary(self):
        fv = FormalVerifier()
        fv.verify_code("x = 1")
        s = fv.summary()
        assert s["total_checks"] > 0
