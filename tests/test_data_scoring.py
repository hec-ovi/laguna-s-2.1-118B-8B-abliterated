"""Sanity for the prompt sets and the stdlib refusal detector."""
import unittest

from laguna_abliterate import data
from laguna_abliterate import scoring


class Data(unittest.TestCase):
    def test_harmful_harmless_matched(self):
        self.assertGreater(len(data.HARMFUL), 0)
        self.assertEqual(len(data.HARMFUL), len(data.HARMLESS))  # paired by index

    def test_lookalike_nonempty_and_disjoint(self):
        self.assertGreater(len(data.LOOKALIKE), 0)
        self.assertTrue(set(data.HARMFUL).isdisjoint(data.HARMLESS))
        self.assertTrue(set(data.HARMFUL).isdisjoint(data.LOOKALIKE))

    def test_split_is_deterministic_and_disjoint(self):
        tr, te = data.split(list(range(10)), train_frac=0.6)
        self.assertEqual(tr, list(range(6)))
        self.assertEqual(te, list(range(6, 10)))
        self.assertEqual(data.split(list(range(10)))[0], tr)  # stable


class Scoring(unittest.TestCase):
    def test_refusal(self):
        self.assertEqual(scoring.classify("I can't help with that request."), "refusal")
        self.assertEqual(scoring.classify("I'm sorry, but I cannot assist."), "refusal")

    def test_compliance(self):
        self.assertEqual(scoring.classify("Here is how you do it. Step 1: open the app."), "compliance")
        self.assertEqual(scoring.classify("```python\ndef f():\n    return 1\n```"), "compliance")

    def test_partial(self):
        # soft refusal that still answers -> partial, not counted as a clean refusal
        self.assertEqual(
            scoring.classify("I'm sorry, but here is a general overview. Step 1: ..."),
            "partial",
        )

    def test_rate_sums_to_one(self):
        texts = ["I can't.", "Here is step 1.", "I'm sorry but here's step 1."]
        r = scoring.refusal_rate(texts)
        self.assertAlmostEqual(r["refusal"] + r["partial"] + r["compliance"], 1.0, places=9)
        self.assertEqual(r["n"], 3)

    def test_mid_answer_disclaimer_not_a_refusal(self):
        text = "Sure, here's the code. Note: I do not condone misuse of this."
        self.assertEqual(scoring.classify(text), "compliance")


if __name__ == "__main__":
    unittest.main()
