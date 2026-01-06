import torch

from ALPACA_DT_Sim.moe_transformer import TransformerWithMoE


def test_infer_observations_and_auxiliary_losses():
    model = TransformerWithMoE(
        input_dim=3,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        out_cont_dim=2,
        out_bin_dim=1,
        num_experts=2,
        num_experts_per_tok=1,
    )
    x = torch.zeros((2, 4, 3), dtype=torch.float32)
    attn_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)

    pred_cont, pred_bin = model(x, attn_mask=attn_mask)
    assert pred_cont is not None and pred_cont.shape == (2, 4, 2)
    assert pred_bin is not None and pred_bin.shape == (2, 4, 1)

    aux = model.get_auxiliary_losses()
    assert "load_balancing_loss" in aux

    y = model.infer_observations(x, cont_idx=[0, 2], bin_idx=[1], out_dim=3, attn_mask=attn_mask)
    assert y.shape == (2, 4, 3)

    # apply_sigmoid=False should emit raw logits in the binary slot
    y_logits = model.infer_observations(
        x, cont_idx=[0, 2], bin_idx=[1], out_dim=3, attn_mask=attn_mask, apply_sigmoid=False
    )
    assert y_logits.shape == (2, 4, 3)
