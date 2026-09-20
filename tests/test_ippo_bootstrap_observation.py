"""Regression for the local critic's post-transition observation selection."""
import ast
from pathlib import Path
import unittest


class BootstrapObservationTest(unittest.TestCase):
    def test_local_bootstrap_uses_post_transition_features(self):
        source = Path(__file__).parents[1] / 'src/peva_sim/train_formal_mappo.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        assignments = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'next_x'
                               for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        # Execute the actual collector expression with distinct successive
        # observations; no simulator or torch dependency is needed here.
        class Args:
            pvf = False
        scope = dict(obs={'time': 0}, next_obs={'time': 30}, cfg=None,
                     args=Args(), features=lambda observation, *_: observation['time'])
        expression = ast.Expression(assignments[0].value)
        result = eval(compile(expression, str(source), 'eval'), scope)
        self.assertEqual(result, 30)


if __name__ == '__main__':
    unittest.main()
