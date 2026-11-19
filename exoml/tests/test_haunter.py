import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import pandas as pd

from exoml.haunter.haunter import Haunter
from exoml.haunter.haunter_generator import INPUT_FEATURES, OUTPUT_TARGETS
from exoml.ml.model.base_model import HyperParams


def simulations(n=400, seed=0):
    rng = np.random.default_rng(seed)
    mass = rng.uniform(1, 600, n)
    pressure = rng.uniform(1, 1000, n)
    temperature = rng.uniform(700, 6000, n)
    return pd.DataFrame({
        'input_mass_Mearth': mass, 'input_CMF': rng.uniform(0, .99, n), 'input_Zenv': rng.uniform(0, 1, n),
        'input_Zwater_core': rng.uniform(0, .5, n), 'input_Tsurf_K': temperature, 'input_Psurf_bar': pressure,
        'flag_physical': True, 'flag_convergence': 0,
        'output_radius_Rearth': 2 + mass ** .3, 'output_entropy_1000bar_SI': 1e3 * (1 + temperature / 100),
        'output_temperature_1000bar_K': temperature * 1.5, 'output_log10_f_s_SI': 31 + np.log10(mass) / 10})


class HaunterTest(unittest.TestCase):
    def setUp(self):
        self.haunter = Haunter(HyperParams(batch_size=64, epochs=1, initial_learning_rate=1e-3), width=16, blocks=1)

    def test_only_converged_physical_rows_are_kept(self):
        data = simulations(10)
        data.loc[0, 'flag_physical'] = False
        data.loc[1, 'flag_convergence'] = 2
        data.loc[2, 'output_radius_Rearth'] = np.nan
        self.assertEqual(len(Haunter.physical_rows(data)), 7)

    def test_wide_columns_are_learned_in_log10(self):
        data = Haunter.add_derived_columns(simulations(5))
        np.testing.assert_allclose(data['input_log10_mass_Mearth'], np.log10(data['input_mass_Mearth']))
        np.testing.assert_allclose(data['output_log10_radius_Rearth'], np.log10(data['output_radius_Rearth']))
        np.testing.assert_allclose(data['output_log10_temperature_1000bar_K'], np.log10(data['output_temperature_1000bar_K']))

    def test_standardization_round_trips(self):
        data = Haunter.add_derived_columns(simulations())
        data = self.haunter.add_standardized_columns(data, data)
        norm = data[[c + '_norm' for c in INPUT_FEATURES + OUTPUT_TARGETS]]
        np.testing.assert_allclose(norm.mean(), 0, atol=1e-6)
        np.testing.assert_allclose(norm.std(ddof=0), 1, atol=1e-6)
        predictions = data[[c + '_norm' for c in OUTPUT_TARGETS]].to_numpy()
        restored = self.haunter.add_destandardized_columns(data.copy(), data, predictions)
        np.testing.assert_allclose(restored['output_radius_Rearth_pred'], data['output_radius_Rearth'], rtol=1e-5)
        np.testing.assert_allclose(restored['output_temperature_1000bar_K_pred'], data['output_temperature_1000bar_K'], rtol=1e-5)

    def test_a_perfect_prediction_has_no_error(self):
        data = Haunter.add_derived_columns(simulations(50))
        data = self.haunter.add_standardized_columns(data, data)
        data = self.haunter.add_destandardized_columns(data, data, data[[c + '_norm' for c in OUTPUT_TARGETS]].to_numpy())
        table = Haunter.evaluate_predictions(data)
        self.assertLess(table['median_rel_%'].max(), 1e-3)

    def test_it_trains_saves_and_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            simulations(600).to_csv(f"{directory}/models_for_ML_training_noduplicates.csv", index=False)
            data = self.haunter.build().load_training_set(training_dir=directory)
            generator = self.haunter.instance_generator(data, None, 64, 6, None)
            loss, _ = self.haunter.instance_loss_accuracy()
            self.haunter.model.compile('adam', loss=loss, metrics=self.haunter.instance_metrics())
            history = self.haunter.model.fit(generator, epochs=2, verbose=0)
            self.assertIn('mae_radius', history.history)
            self.haunter.model.save(f"{directory}/m.keras")
            reloaded = Haunter(self.haunter.hyperparams, width=16, blocks=1).load_model(f"{directory}/m.keras")
            self.assertEqual(reloaded.model.output_shape, (None, 4))


if __name__ == '__main__':
    unittest.main()
