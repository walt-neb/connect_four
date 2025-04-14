# filename: disassemble_checkpoint.py
"""
Tool to extract individual agent model weights from a combined training checkpoint file.

Usage:
  python disassemble_checkpoint.py <path_to_checkpoint.pth>

This script reads a checkpoint file created by 'train_c4.py' (which contains
episode number, optimizer states, and model state dictionaries) and saves
the 'agent1_state_dict' and 'agent2_state_dict' into separate .pth files
named appropriately (e.g., m1_basename_epXXX.pth, m2_basename_epXXX.pth).

These individual weight files can then be used by 'play_human.py' or
'play_two_models.py'.
"""

import torch
import os
import sys
import re
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Extract individual agent weights (state_dicts) from a combined training checkpoint file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'checkpoint_path',
        type=str,
        help="Path to the combined checkpoint file (e.g., ./wts/checkpoint_h7_ep22000.pth)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./wts', # Default to saving in the same directory as checkpoints
        help="Directory where the extracted weight files (m1_*.pth, m2_*.pth) will be saved."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir

    # --- Validate Input Path ---
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        sys.exit(1)
    if not os.path.isfile(checkpoint_path):
         print(f"Error: Provided path '{checkpoint_path}' is not a file.")
         sys.exit(1)

    # --- Create Output Directory if Needed ---
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist. Creating it.")
        os.makedirs(output_dir)

    # --- Load the Checkpoint ---
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        # Load onto CPU initially to avoid GPU memory issues if just extracting
        # Use weights_only=False as checkpoint contains optimizer states etc.
        # Ensure you trust the checkpoint file source.
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        sys.exit(1)

    # --- Validate Checkpoint Contents ---
    required_keys = ['agent1_state_dict', 'agent2_state_dict', 'episode']
    if not all(key in checkpoint for key in required_keys):
        print("Error: Checkpoint file is missing required keys.")
        print(f"Expected keys like: {required_keys}")
        print(f"Found keys: {list(checkpoint.keys())}")
        sys.exit(1)

    # --- Extract Information ---
    agent1_state_dict = checkpoint['agent1_state_dict']
    agent2_state_dict = checkpoint['agent2_state_dict']
    # Episode number saved is usually the *next* episode to start
    # The weights correspond to the *previous* episode number
    episode_saved = checkpoint['episode'] - 1

    # --- Determine Output Filenames ---
    checkpoint_filename = os.path.basename(checkpoint_path)
    # Try to extract base name and episode number from checkpoint filename
    # e.g., checkpoint_h7_ep22000.pth -> base='h7', episode=22000
    match = re.match(r"^checkpoint_(.*?)_ep(\d+)\.pth$", checkpoint_filename)
    if match:
        hyp_base_name = match.group(1)
        ep_from_filename = int(match.group(2))
        # Consistency check
        if ep_from_filename != episode_saved:
            print(f"Warning: Episode number in filename ({ep_from_filename}) does not match episode saved in checkpoint ({episode_saved}). Using episode from checkpoint.")
        episode_num_for_name = episode_saved
    else:
        # Fallback if filename pattern doesn't match - use base from hyp and ep from ckpt
        print("Warning: Could not parse base name and episode from checkpoint filename. Using generic name.")
        hyp_base_name = "extracted" # Generic fallback
        episode_num_for_name = episode_saved

    # Construct output paths
    m1_filename = f"m1_{hyp_base_name}_ep{episode_num_for_name}.pth"
    m2_filename = f"m2_{hyp_base_name}_ep{episode_num_for_name}.pth"
    m1_output_path = os.path.join(output_dir, m1_filename)
    m2_output_path = os.path.join(output_dir, m2_filename)

    # --- Save Individual Weights ---
    try:
        torch.save(agent1_state_dict, m1_output_path)
        print(f"Successfully extracted Agent 1 weights to: {m1_output_path}")
    except Exception as e:
        print(f"Error saving Agent 1 weights to {m1_output_path}: {e}")

    try:
        torch.save(agent2_state_dict, m2_output_path)
        print(f"Successfully extracted Agent 2 weights to: {m2_output_path}")
    except Exception as e:
        print(f"Error saving Agent 2 weights to {m2_output_path}: {e}")

if __name__ == "__main__":
    main()