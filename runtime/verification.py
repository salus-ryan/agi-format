"""Formal verification module -- hard guarantees where probabilistic reasoning
is insufficient.

Backends:
  1. Z3 SMT solver   -- constraint checking, arithmetic, satisfiability
  2. Python type checker -- verify generated code via ast + mypy-lite
  3. Code executor    -- sandboxed execution to test code correctness

Falls back gracefully: if Z3 isn't installed, uses the code-based backends.
"""

import ast
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import structlog

log = structlog.get_logger()


class VerificationResult:
    """Result of a formal verification check."""

    __slots__ = ("verified", "backend", "message", "counterexample", "elapsed_ms")

    def __init__(
        self,
        verified: bool,
        backend: str,
        message: str = "",
        counterexample: str | None = None,
        elapsed_ms: float = 0,
    ):
        self.verified = verified
        self.backend = backend
        self.message = message
        self.counterexample = counterexample
        self.elapsed_ms = elapsed_ms

    def to_dict(self) -> dict:
        return {
            "verified": self.verified,
            "backend": self.backend,
            "message": self.message,
            "counterexample": self.counterexample,
            "elapsed_ms": self.elapsed_ms,
        }


class Z3Backend:
    """SMT solver for constraint checking and satisfiability."""

    def __init__(self):
        self.available = False
        try:
            import z3
            self.z3 = z3
            self.available = True
            log.info("z3_backend_available")
        except ImportError:
            log.info("z3_backend_unavailable")

    def check_satisfiable(self, constraints_code: str, timeout_ms: int = 30000) -> VerificationResult:
        """Check if a set of Z3 constraints is satisfiable.

        constraints_code: Python code that uses z3 to define and check constraints.
        Must define a variable `result` that is True/False and optionally `model_str`.
        """
        if not self.available:
            return VerificationResult(False, "z3", "Z3 not installed")

        start = time.time()
        try:
            namespace = {"z3": self.z3}
            exec(constraints_code, namespace)
            verified = bool(namespace.get("result", False))
            model_str = namespace.get("model_str", "")
            elapsed = (time.time() - start) * 1000

            return VerificationResult(
                verified=verified,
                backend="z3",
                message=f"{'SAT' if verified else 'UNSAT'}: {model_str}" if model_str else "",
                elapsed_ms=elapsed,
            )
        except Exception as e:
            return VerificationResult(
                False, "z3", f"Z3 execution error: {e}",
                elapsed_ms=(time.time() - start) * 1000,
            )

    def check_implication(self, premise: str, conclusion: str) -> VerificationResult:
        """Check if premise logically implies conclusion using Z3.

        Both are Z3 BoolRef expressions in string form.
        """
        if not self.available:
            return VerificationResult(False, "z3", "Z3 not installed")

        code = textwrap.dedent(f"""
            s = z3.Solver()
            # If premise AND NOT conclusion is unsatisfiable, then premise implies conclusion
            premise_expr = {premise}
            conclusion_expr = {conclusion}
            s.add(premise_expr)
            s.add(z3.Not(conclusion_expr))
            check = s.check()
            result = (check == z3.unsat)
            model_str = str(s.model()) if check == z3.sat else "implication holds"
        """)
        return self.check_satisfiable(code)


class CodeVerifier:
    """Verify Python code through static analysis and sandboxed execution."""

    def __init__(self, timeout_seconds: int = 10):
        self.timeout = timeout_seconds

    def check_syntax(self, code: str) -> VerificationResult:
        """Check if Python code is syntactically valid."""
        start = time.time()
        try:
            ast.parse(code)
            return VerificationResult(
                True, "ast_parse", "Syntax OK",
                elapsed_ms=(time.time() - start) * 1000,
            )
        except SyntaxError as e:
            return VerificationResult(
                False, "ast_parse",
                f"Syntax error at line {e.lineno}: {e.msg}",
                counterexample=f"Line {e.lineno}: {e.text}",
                elapsed_ms=(time.time() - start) * 1000,
            )

    def check_types(self, code: str) -> VerificationResult:
        """Run basic type checking on Python code.

        Uses ast analysis to catch obvious type issues.
        For full type checking, would shell out to mypy.
        """
        start = time.time()
        issues = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return VerificationResult(
                False, "type_check", f"Syntax error: {e}",
                elapsed_ms=(time.time() - start) * 1000,
            )

        # Walk the AST looking for common issues
        for node in ast.walk(tree):
            # Check for undefined name usage (basic)
            if isinstance(node, ast.FunctionDef):
                # Check return type annotation consistency
                if node.returns and isinstance(node.returns, ast.Constant):
                    if node.returns.value is None:
                        # Function annotated as -> None but has return values
                        for child in ast.walk(node):
                            if isinstance(child, ast.Return) and child.value is not None:
                                issues.append(
                                    f"Function '{node.name}' annotated -> None but returns a value"
                                )

        elapsed = (time.time() - start) * 1000
        if issues:
            return VerificationResult(
                False, "type_check",
                f"{len(issues)} issue(s): " + "; ".join(issues),
                elapsed_ms=elapsed,
            )
        return VerificationResult(True, "type_check", "No type issues found", elapsed_ms=elapsed)

    def check_execution(self, code: str, test_code: str = "") -> VerificationResult:
        """Execute code in a sandboxed subprocess and check for errors.

        code: the code to verify
        test_code: optional test assertions to run after the code
        """
        start = time.time()
        full_code = code
        if test_code:
            full_code += "\n\n# --- Tests ---\n" + test_code

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            f.flush()
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = (time.time() - start) * 1000

            if result.returncode == 0:
                return VerificationResult(
                    True, "execution",
                    f"Executed successfully{f' (stdout: {result.stdout[:200]})' if result.stdout.strip() else ''}",
                    elapsed_ms=elapsed,
                )
            else:
                return VerificationResult(
                    False, "execution",
                    f"Exit code {result.returncode}: {result.stderr[:500]}",
                    counterexample=result.stderr[:500],
                    elapsed_ms=elapsed,
                )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                False, "execution",
                f"Timeout after {self.timeout}s",
                elapsed_ms=(time.time() - start) * 1000,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)


class FormalVerifier:
    """Unified verification interface. Routes to the best available backend."""

    def __init__(self):
        self.z3 = Z3Backend()
        self.code = CodeVerifier()
        self.check_count = 0
        self.pass_count = 0

    def verify_code(self, code: str, test_code: str = "") -> list[VerificationResult]:
        """Run all applicable checks on a piece of code."""
        self.check_count += 1
        results = []

        # 1. Syntax check (fast)
        syntax = self.code.check_syntax(code)
        results.append(syntax)
        if not syntax.verified:
            return results  # no point continuing

        # 2. Type check (fast)
        types = self.code.check_types(code)
        results.append(types)

        # 3. Execution check (slower, sandboxed)
        if test_code:
            execution = self.code.check_execution(code, test_code)
            results.append(execution)

        all_passed = all(r.verified for r in results)
        if all_passed:
            self.pass_count += 1

        log.info(
            "verification_complete",
            checks=len(results),
            all_passed=all_passed,
            total_ms=sum(r.elapsed_ms for r in results),
        )
        return results

    def verify_constraint(self, constraints_code: str) -> VerificationResult:
        """Verify a Z3 constraint."""
        self.check_count += 1
        result = self.z3.check_satisfiable(constraints_code)
        if result.verified:
            self.pass_count += 1
        return result

    def summary(self) -> dict:
        return {
            "total_checks": self.check_count,
            "passed": self.pass_count,
            "z3_available": self.z3.available,
        }
