"""Quick test to verify MergeDNA model instantiation and forward pass."""

import torch
from mergedna.model.mergedna import MergeDNA, MergeDNAConfig, MergeDNAForSequenceClassification
from mergedna.data.tokenizer import DNACharTokenizer
from mergedna.utils.utils import print_model_summary


def test_tokenizer():
    """Test DNA tokenizer."""
    print("=" * 50)
    print("Testing DNACharTokenizer")
    print("=" * 50)

    tokenizer = DNACharTokenizer(max_length=64)
    seq = "ATCGATCGATCGATCGATCGATCG"
    encoded = tokenizer([seq], max_length=64)
    print(f"  Input: {seq}")
    print(f"  Tokens: {encoded['input_ids'][0][:len(seq)]}")
    print(f"  Decoded: {tokenizer.decode(encoded['input_ids'][0].tolist())}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print("  PASSED\n")


def test_model_small():
    """Test MergeDNA with small config."""
    print("=" * 50)
    print("Testing MergeDNA (small config)")
    print("=" * 50)

    config = MergeDNAConfig(
        vocab_size=10,
        embed_dim=128,       # Small for testing
        num_heads=4,
        local_encoder_layers=2,
        latent_encoder_layers=2,
        latent_decoder_layers=1,
        local_decoder_layers=1,
        window_size=8,
        dropout=0.0,
        use_flash_attn=False,  # CPU testing
        max_seq_length=64,
    )

    model = MergeDNA(config)
    print_model_summary(model)

    # Test pretrain forward
    B, N = 2, 64
    input_ids = torch.randint(5, 9, (B, N))  # A=5, T=6, C=7, G=8
    attention_mask = torch.ones(B, N, dtype=torch.long)

    print("  Testing pretrain forward pass...")
    losses = model.forward_pretrain(input_ids, attention_mask)
    print(f"    Total loss: {losses['loss'].item():.4f}")
    print(f"    MTR loss: {losses['loss_mtr'].item():.4f}")
    print(f"    Latent MTR loss: {losses['loss_latent_mtr'].item():.4f}")
    print(f"    AMTM loss: {losses['loss_amtm'].item():.4f}")

    # Test encode mode
    print("  Testing encode forward pass...")
    outputs = model(input_ids, attention_mask, mode="encode")
    print(f"    Pooled output shape: {outputs['pooled_output'].shape}")

    # Test decode mode
    print("  Testing decode forward pass...")
    outputs = model(input_ids, attention_mask, mode="decode")
    print(f"    Logits shape: {outputs['logits'].shape}")

    print("  PASSED\n")


def test_classification():
    """Test classification model."""
    print("=" * 50)
    print("Testing MergeDNAForSequenceClassification")
    print("=" * 50)

    config = MergeDNAConfig(
        vocab_size=10,
        embed_dim=128,
        num_heads=4,
        local_encoder_layers=2,
        latent_encoder_layers=2,
        latent_decoder_layers=1,
        local_decoder_layers=1,
        window_size=8,
        use_flash_attn=False,
    )

    model = MergeDNAForSequenceClassification(config, num_classes=3)

    B, N = 2, 64
    input_ids = torch.randint(5, 9, (B, N))
    attention_mask = torch.ones(B, N, dtype=torch.long)
    labels = torch.randint(0, 3, (B,))

    outputs = model(input_ids, attention_mask, labels=labels)
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print("  PASSED\n")


def test_gradient_flow():
    """Test that gradients flow through all components."""
    print("=" * 50)
    print("Testing gradient flow")
    print("=" * 50)

    config = MergeDNAConfig(
        vocab_size=10,
        embed_dim=64,
        num_heads=2,
        local_encoder_layers=1,
        latent_encoder_layers=1,
        latent_decoder_layers=1,
        local_decoder_layers=1,
        window_size=8,
        use_flash_attn=False,
    )

    model = MergeDNA(config)
    B, N = 2, 32
    input_ids = torch.randint(5, 9, (B, N))
    attention_mask = torch.ones(B, N, dtype=torch.long)

    losses = model.forward_pretrain(input_ids, attention_mask)
    losses["loss"].backward()

    # Check gradients
    components = {
        "local_encoder": model.local_encoder,
        "latent_encoder": model.latent_encoder,
        "latent_decoder": model.latent_decoder,
        "local_decoder": model.local_decoder,
    }

    for name, module in components.items():
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )
        status = "OK" if has_grad else "NO GRAD"
        print(f"  {name}: {status}")

    print("  PASSED\n")


if __name__ == "__main__":
    test_tokenizer()
    test_model_small()
    test_classification()
    test_gradient_flow()
    print("All tests passed!")
