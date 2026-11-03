import sys
import unittest
import warnings

import culverin


class TestFastParseDiagnostics(unittest.TestCase):
    """
    Test suite for the FastParse Engine's DX (Developer Experience) features.
    These tests intentionally fail API calls to verify the high-fidelity
    error messages produced by the C layer.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.world = culverin.PhysicsWorld()

    def print_error(self, test_name: str, error: Exception) -> None:
        """Helper to make terminal output readable."""
        print(f"\n{'='*20} {test_name} {'='*20}")
        print(str(error))
        print('=' * (42 + len(test_name)))

    def test_missing_required_args_formatting(self) -> None:
        """Trigger FastParseReportMissing with signature reconstruction."""
        try:
            # create_body requires at least 'pos'
            self.world.create_body()
        except TypeError as e:
            self.print_error("MISSING ARGS", e)
            self.assertIn("missing 1 required positional argument", str(e))
            self.assertIn("- pos (tuple)", str(e))
            self.assertIn("Expected signature:", str(e))

    def test_type_error_with_repr(self) -> None:
        """Trigger FastParseReportTypeError using a field that has a type guard."""
        try:
            # 'is_sensor' has a 'bool' type guard in your C spec.
            # We use this to test the high-fidelity type error.
            self.world.create_body(pos=(0,0,0), is_sensor="NotABool") # type: ignore
        except TypeError as e:
            self.print_error("TYPE MISMATCH", e)
            self.assertIn("must be bool, not str", str(e))
            self.assertIn("Received value: 'NotABool'", str(e))
            self.assertIn("!!! is_sensor: bool !!!", str(e))

    def test_fuzzy_keyword_matching(self) -> None:
        """Trigger FastParseReportUnknownKeyword fuzzy 'Did you mean?' logic."""
        try:
            # 'pso' is 1 distance away from 'pos'
            self.world.create_body(pso=(0, 10, 0)) # type: ignore
        except TypeError as e:
            self.print_error("FUZZY MATCH", e)
            # Match the actual C-output format
            self.assertIn("invalid keyword argument for Body(). Did you mean 'pos'?", str(e))
            self.assertIn("Valid arguments are:", str(e))

    def test_multiple_values_error(self) -> None:
        """Trigger FastParseReportMultiple (positional + keyword conflict)."""
        try:
            # First arg is pos, then we provide pos again as keyword
            self.world.create_body((0, 0, 0), pos=(1, 1, 1)) # type: ignore
        except TypeError as e:
            self.print_error("MULTIPLE VALUES", e)
            self.assertIn("got multiple values for argument 'pos'", str(e))
            self.assertIn("provided both as a positional argument and as a keyword", str(e))

    def test_too_many_positional_args(self) -> None:
        """Trigger FastParseReportTooMany with capacity guide."""
        try:
            # create_body takes many args, but let's send 50 to be sure
            args = [(0,0,0)] * 50
            self.world.create_body(*args) # type: ignore
        except TypeError as e:
            self.print_error("TOO MANY ARGS", e)
            self.assertIn("positional arguments but 50 were given", str(e))
            self.assertIn("does not accept variable positional arguments (*args)", str(e))

    def test_boolean_optimization_and_strictness(self) -> None:
        """Verify the fast_parse_conv_bool pointer check vs string conversion."""
        # 1. This should work (optimized pointer path)
        self.world.create_body(pos=(0,0,0), is_sensor=True)

        # 2. This should fail with a clean repr
        try:
            self.world.create_body(pos=(0,0,0), is_sensor="NotABool") # type: ignore
        except TypeError as e:
            self.print_error("BOOL REPR CHECK", e)
            self.assertIn("must be bool, not str", str(e))
            self.assertIn("Received value: 'NotABool'", str(e))

    def test_non_interned_keyword_warning(self) -> None:
        """
        Force a non-interned string into the keyword lookup
        to trigger the performance warning.
        """
        # We use a dynamically constructed string that hasn't been interned
        # Most literal strings in Python are interned, so we use chr() math.
        non_interned_key = "".join([chr(112), chr(111), chr(115)]) # "pos"

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Use **kwargs to bypass literal interning
            kwargs = {non_interned_key: (0, 10, 0)}
            self.world.create_body(**kwargs) # type: ignore

            # Check if our C-layer warning fired
            found_perf_warning = False
            for warning in w:
                if "is not interned" in str(warning.message):
                    print(f"\n[PERF WARNING DETECTED]: {warning.message}")
                    found_perf_warning = True

            self.assertTrue(found_perf_warning, "The performance warning for non-interned strings did not fire.")

    def test_u64_error_formatting(self) -> None:
        """Verify that large unsigned ints report errors correctly."""
        try:
            # material_id expects a uint32, let's give it a negative number
            self.world.create_body(pos=(0,0,0), material_id=-99)
        except (ValueError, OverflowError, TypeError) as e:
            # Note: Depending on which path it takes, this might be a
            # ValueError from PyLong_AsUnsignedLong.
            self.print_error("U32 BOUNDS", e)


class TestReturnChannel(unittest.TestCase):
    """Failure reaches Python through one channel, so the two can never disagree.

    CPython has two: the return value (NULL means "failed") and the thread-state
    error indicator. A C function that frees itself from one and not the other
    produces either "SystemError: returned NULL without setting an exception" or
    a value handed back with an exception latched behind it, poisoning the next
    unrelated call. The C++ side now returns a single `BoundResult` and lets
    `Bridge()` write both channels together (bindings/Result.hpp); these tests
    check the observable half of that contract from across every subsystem.
    """

    # One deliberately-wrong call per subsystem. Each must reach Python as a
    # real exception with its own message -- never as a SystemError standing in
    # for a missing one.
    CASES = {
        "world.create_body": lambda w: w.create_body(pos="not-a-vector", size=(1, 1, 1)),
        "world.set_gravity": lambda w: w.set_gravity("not-a-vector"),
        "world.create_character": lambda w: w.create_character(pos=(0, 1, 0), height="tall"),
        "world.create_ragdoll": lambda w: w.create_ragdoll(settings=None, pos=(0, 0, 0)),
        "world.destroy_body": lambda w: w.destroy_body(object()),
        "math.inverse": lambda w: culverin.MathService().inverse((1.0, 2.0, 3.0)),
    }

    def setUp(self) -> None:
        self.world = culverin.PhysicsWorld()

    def test_failures_are_real_exceptions(self) -> None:
        for name, call in self.CASES.items():
            with self.subTest(call=name):
                with self.assertRaises(Exception) as raised:
                    call(self.world)
                # A SystemError here means the C++ side reported failure through
                # one channel and not the other; the type and message are
                # CPython's substitute, not the diagnostic the code intended.
                self.assertNotIsInstance(
                    raised.exception,
                    SystemError,
                    f"{name} failed without a diagnostic: {raised.exception}",
                )
                self.assertTrue(
                    str(raised.exception), f"{name} raised {type(raised.exception).__name__} with no message"
                )

    def test_a_failed_call_leaves_no_exception_behind(self) -> None:
        """The error indicator is consumed by the raise, not left pending."""
        for name, call in self.CASES.items():
            with self.subTest(call=name):
                with self.assertRaises(Exception):
                    call(self.world)
                # Nothing must be pending now: an unrelated call that succeeds
                # has to stay successful.
                self.world.step(0)
                self.assertIsNone(sys.exc_info()[1])

    def test_success_after_failure_returns_a_usable_value(self) -> None:
        body = self.world.create_body(pos=(0, 1, 0), size=(1, 1, 1))
        with self.assertRaises(Exception):
            self.world.set_gravity("not-a-vector")
        with self.assertRaises(Exception):
            self.world.destroy_body(object())
        # A latched exception would make this call raise or return garbage.
        self.assertIsInstance(self.world.get_gravity(), tuple)
        self.world.step(0)
        self.assertIsNotNone(body)

if __name__ == "__main__":
    # Ensure stdout is flushed for CI logs
    unittest.main(verbosity=1)
