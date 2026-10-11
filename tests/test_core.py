import unittest
from pathlib import Path
import numpy as np
from seasonal_tree.sampler import make_model
from seasonal_tree.data import load_public, build_W, validate_public
from seasonal_tree.tree_base import seeded_split
from seasonal_tree.likelihood import likelihood, q_tip
from seasonal_tree.mentor import collapsed_map
from seasonal_tree.proposals import propose
from seasonal_tree.validation import ExactToy

PUBLIC = Path(__file__).resolve().parents[1] / 'examples/five_tip/public.npz'

class CoreChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_model(PUBLIC)

    def test_all_single_region_origins(self):
        m = self.model
        for r in range(27):
            np.testing.assert_array_equal(m.root(1 << r), m.K * (m.data['state_blocks'] == r))
        for r in (-1, 0, 3, 7, 1 << 27):
            with self.assertRaises(ValueError): m.root(r)

    def test_mass_preserved_at_split(self):
        m = self.model
        n = m.K * ((np.arange(len(m.K)) % 3) == 0)
        a, b = seeded_split(n, 127, m.data['cells'], m.neighbours, m.order)
        np.testing.assert_array_equal(a + b, n)
        self.assertGreater(a.sum(), 0)
        self.assertGreater(b.sum(), 0)

    def test_fixed_two_offspring_formula(self):
        k = np.array([25.]); phi = np.array([.2]); w = .98 * (1 - 1e-7)
        gamma = np.array([1 - (1 - w / k[0]) ** 2])
        got = collapsed_map(phi, k, np.array([0, 1]), np.array([0]), gamma)
        expected = 1 - (1 - gamma[0] * phi[0]) ** k[0]
        self.assertAlmostEqual(got[0], expected, places=13)

    def test_incompatible_and_off_map_arrivals_are_lost(self):
        data = {'cells': np.array([[0, 0], [0, 1]]), 'state_genotypes': np.array([0, 1])}
        w = build_W(data).toarray()
        self.assertAlmostEqual(w[1, 0], .005 * 1e-7)
        self.assertAlmostEqual(w[0, 1], .005 * 1e-7 / 2)
        self.assertTrue(np.all(w.sum(0) < 1))

    def test_unfloored_likelihood(self):
        q = np.array([[.2, .8, 0.]])
        c = np.array([[2, 3, 0]])
        self.assertAlmostEqual(likelihood(q, c), 2 * np.log(.2) + 3 * np.log(.8))
        self.assertEqual(likelihood(q, np.array([[0, 0, 1]])), -np.inf)
        self.assertTrue(np.isfinite(likelihood(q, np.array([[0, 0, 1]]), .001)))

    def test_shape_error_detected_before_jit(self):
        d = load_public(PUBLIC); d['times'] = np.zeros(8, int)
        with self.assertRaises(ValueError): validate_public(d)

    def test_prefix_cache_is_full_replay(self):
        m = self.model
        with np.load(PUBLIC.parent / 'initialization.npz') as z:
            r = int(z['masks'][0]); u = z['seeds'][0].copy()
        h = m.run_noise(r, u)
        self.assertEqual(h[3], 0)
        edge = 0; begin = int(m.data['times'][edge]) - 5
        v = u.copy(); v[m.offset[edge] + begin] = 2026101177
        partial = m.run_noise(r, v, h, (edge, begin)); fresh = m.run_noise(r, v)
        self.assertEqual(partial[3], fresh[3])
        for a, b in zip(partial[:3], fresh[:3]): np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(partial[0][1:], h[0][1:])

    def test_known_root_proposals_keep_location(self):
        m = ExactToy(); rng = np.random.default_rng(11)
        r = 1 << 9; u = rng.integers(2**31 - 1, size=12); h = m.run_noise(r, u)
        counts = np.zeros((5, 27), int); counts[0, 8] = 5
        for _ in range(100):
            p, _ = propose(m, r, u, h, rng, 'known', counts, 0.)
            if p is not None: self.assertEqual(p[0], r)

if __name__ == '__main__': unittest.main()
