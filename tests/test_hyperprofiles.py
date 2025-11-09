from ncps.neurons import load_profile


def test_load_tf_profile():
    profile = load_profile("ltcse_tf")
    assert profile.name == "tensorflow_legacy"
    assert profile.ode_unfolds == 6
    assert profile.constraints["w_min"] == 1e-5
