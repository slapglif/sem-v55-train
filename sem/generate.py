"""Text generation with SEM V5.5 'Lean Crystal'.

Loads a trained model checkpoint and generates text using
Born Collapse sampling.
"""
import torch
from pathlib import Path
from typing import Optional

from .config import SEMConfig
from .model import SEMModel


def load_model(checkpoint_path: str, device: str = 'cpu') -> SEMModel:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device

    Returns:
        Loaded SEMModel
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = SEMConfig()

    model = SEMModel(config).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def generate(model: SEMModel, prompt_ids: torch.Tensor,
             max_new_tokens: int = 100,
             temperature: float = 1.0,
             top_k: int = 50,
             top_p: float = 0.95,
             device: str = 'cpu') -> torch.Tensor:
    """Generate tokens from a prompt.

    Args:
        model: Trained SEMModel
        prompt_ids: [1, S] prompt token indices
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus filtering
        device: Device

    Returns:
        [1, S + max_new_tokens] generated tokens
    """
    prompt_ids = prompt_ids.to(device)
    return model.generate(
        prompt_ids, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate with SEM V5.5')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default='0,1,2,3')
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)

    # Parse prompt as comma-separated token IDs
    prompt = torch.tensor([[int(x) for x in args.prompt.split(',')]])

    output = generate(model, prompt, args.max_tokens, args.temperature, device=args.device)
    print(f"Generated tokens: {output[0].tolist()}")
