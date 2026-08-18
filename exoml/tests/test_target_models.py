import numpy as np
import pandas as pd
import pytest

from exoml.haunter.haunter import Haunter, StandardizeScaler
from exoml.haunter.haunter_generator import HaunterModelGenerator
from exoml.iatson.IATSON_planet import IATSON_planet
from exoml.ml.model.base_model import HyperParams
from exoml.santo.SANTO import SANTO, clean_to_float, weighted_binary_crossentropy_param


class _Dataset:
    def batch(self, batch_size):
        return self


class _PredictionModel:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=np.float32)
        self.calls = []

    def predict(self, dataset, batch_size, steps):
        self.calls.append((dataset, batch_size, steps))
        return self.predictions


class _IatsonInput:
    def __init__(self, length=2):
        self.dataset = _Dataset()
        self.length = length

    def get_dataset(self):
        return self.dataset

    def __len__(self):
        return self.length


def _iatson():
    return IATSON_planet(
        class_ids={0: "planet"},
        class_names={0: "planet"},
        type_to_label={"planet": 0},
        hyperparams=HyperParams(),
    )


def test_iatson_planet_predict_batch_shape_and_thresholding():
    model = _PredictionModel([[0.4], [0.8]])
    iatson = _iatson()
    iatson.model = model

    indexes, values = iatson.predict_batch(_IatsonInput(), batch_size=2)

    np.testing.assert_array_equal(indexes, [-1, 0])
    np.testing.assert_allclose(values, [-1.0, 0.8])
    assert values.shape == (2,)
    assert model.calls[0][1:] == (2, 2)


def test_iatson_planet_prediction_threshold_rejects_malformed_predictions():
    with pytest.raises((IndexError, TypeError, ValueError)):
        _iatson().predict_threshold(np.array([0.2, 0.8]))


def test_haunter_mock_model_prediction_is_finite_and_batched():
    haunter = Haunter(HyperParams(dropout_rate=0.0))
    model = _PredictionModel(np.full((3, 4), 0.25, dtype=np.float32))
    haunter.model = model
    inputs = np.arange(18, dtype=np.float32).reshape(3, 6)

    predictions = model.predict(inputs, batch_size=2, steps=2)

    assert predictions.shape == (3, 4)
    assert np.isfinite(predictions).all()


def test_haunter_generator_handles_partial_batch():
    columns = {
        "input_mass_Mearth_norm": [0.1, 0.2, 0.3],
        "input_CMF_norm": [0.1, 0.2, 0.3],
        "input_Zenv_norm": [0.1, 0.2, 0.3],
        "input_Zwater_core_norm": [0.1, 0.2, 0.3],
        "input_Tsurf_K_norm": [0.1, 0.2, 0.3],
        "input_Psurf_bar_norm": [0.1, 0.2, 0.3],
        "output_radius_Rearth_norm": [0.1, 0.2, 0.3],
        "output_entropy_log10_1000bar_SI_norm": [0.1, 0.2, 0.3],
        "output_log10_f_s_SI_norm": [0.1, 0.2, 0.3],
        "output_log10_temperature_1000bar_K_norm": [0.1, 0.2, 0.3],
    }
    generator = HaunterModelGenerator(pd.DataFrame(columns), batch_size=2, input_size=6, shuffle=False)

    assert len(generator) == 2
    first_x, first_y = generator[0]
    second_x, second_y = generator[1]
    assert first_x.shape == (2, 6)
    assert first_y.shape == (2, 4)
    assert second_x.shape == (1, 6)
    assert second_y.shape == (1, 4)


def test_haunter_generator_rejects_missing_input_column():
    with pytest.raises(KeyError):
        HaunterModelGenerator(pd.DataFrame({"input_mass_Mearth_norm": [1.0]}), 1, 6, shuffle=False)[0]


def test_haunter_standardize_scaler_round_trip_is_finite():
    scaler = StandardizeScaler()
    values = np.array([1.0, 2.0, 3.0])
    scaler.fit(values)

    transformed = scaler.transform(values)
    restored = scaler.inverse_transform(transformed)

    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(restored, values)


def test_santo_weighted_loss_factory_returns_callable():
    assert callable(weighted_binary_crossentropy_param(weight=2.0))


def test_santo_gap_filling_and_autocorrelation_outputs():
    santo = SANTO()
    times = np.array([0.0, 1.0 / 1440.0, 3.0 / 1440.0, 4.0 / 1440.0])
    data = np.array([1.0, 0.5, 1.0, 0.8])

    filled_times, filled_data, cadence = santo.fill_data_gaps(times, data)
    lags, correlation = santo.compute_autocorrelation(filled_data, max_days=1, cadence_s=120)

    assert len(filled_times) == len(filled_data) == 5
    assert np.isclose(filled_data[2], 1e-7)
    assert cadence > 0
    assert lags.shape == correlation.shape
    assert np.isfinite(correlation).all()
    assert np.isclose(correlation[0], 1.0)


def test_santo_sde_is_finite_for_nonconstant_signal():
    result = SANTO().compute_sde(np.array([0.1, 0.4, 0.2, 0.8, 0.3, 0.6]), window_size=3)

    assert result.shape == (6,)
    assert np.isfinite(result).all()


def test_santo_clean_to_float_rejects_malformed_input():
    with pytest.raises(ValueError):
        clean_to_float("not-a-number")
