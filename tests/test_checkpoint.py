import torch
from safetensors.torch import save_file

from feelwm.checkpoint import read_checkpoint


def test_pt_and_safetensors_give_the_same_weights(tmp_path):
    state = {"a.weight": torch.randn(3, 4), "a.bias": torch.randn(4)}
    torch.save({"model_state": state, "next_epoch": 2}, tmp_path / "m.pt")
    save_file(state, tmp_path / "m.safetensors")

    pt = read_checkpoint(tmp_path / "m.pt")
    st = read_checkpoint(str(tmp_path / "m.safetensors"), map_location=torch.device("cpu"))
    assert pt["next_epoch"] == 2
    assert list(st) == ["model_state"]
    assert set(st["model_state"]) == set(pt["model_state"]) == set(state)
    for k, v in state.items():
        assert torch.equal(pt["model_state"][k], v)
        assert torch.equal(st["model_state"][k], v)
