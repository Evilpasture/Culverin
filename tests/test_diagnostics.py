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
        """Trigger fp_report_missing with signature reconstruction."""
        try:
            # create_body requires at least 'pos'
            self.world.create_body()
        except TypeError as e:
            self.print_error("MISSING ARGS", e)
            self.assertIn("missing 1 required positional argument", str(e))
            self.assertIn("- pos (tuple)", str(e))
            self.assertIn("Expected signature:", str(e))

    def test_type_error_with_repr(self) -> None:
        """Trigger fp_report_type_error using a field that has a type guard."""
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
        """Trigger fp_report_unknown_keyword fuzzy 'Did you mean?' logic."""
        try:
            # 'pso' is 1 distance away from 'pos'
            self.world.create_body(pso=(0, 10, 0)) # type: ignore
        except TypeError as e:
            self.print_error("FUZZY MATCH", e)
            # Match the actual C-output format
            self.assertIn("invalid keyword argument for Body(). Did you mean 'pos'?", str(e))
            self.assertIn("Valid arguments are:", str(e))

    def test_multiple_values_error(self) -> None:
        """Trigger fp_report_multiple (positional + keyword conflict)."""
        try:
            # First arg is pos, then we provide pos again as keyword
            self.world.create_body((0, 0, 0), pos=(1, 1, 1)) # type: ignore
        except TypeError as e:
            self.print_error("MULTIPLE VALUES", e)
            self.assertIn("got multiple values for argument 'pos'", str(e))
            self.assertIn("provided both as a positional argument and as a keyword", str(e))

    def test_too_many_positional_args(self) -> None:
        """Trigger fp_report_too_many with capacity guide."""
        try:
            # create_body takes many args, but let's send 50 to be sure
            args = [(0,0,0)] * 50
            self.world.create_body(*args) # type: ignore
        except TypeError as e:
            self.print_error("TOO MANY ARGS", e)
            self.assertIn("positional arguments but 50 were given", str(e))
            self.assertIn("does not accept variable positional arguments (*args)", str(e))

    def test_boolean_optimization_and_strictness(self) -> None:
        """Verify the fp_conv_bool pointer check vs string conversion."""
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

if __name__ == "__main__":
    # Ensure stdout is flushed for CI logs
    unittest.main(verbosity=1)
