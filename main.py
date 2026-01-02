"""
Main entry point for AutoAUC.
"""

import argparse
import sys
from src.ui import cli_main, gui_main


def main():
    """Main entry point with interface selection."""
    parser = argparse.ArgumentParser(description='AutoAUC - Automated AUC Optimization')
    parser.add_argument('--interface', choices=['cli', 'gui'], default='gui',
                       help='Interface to use (default: gui)')
    
    args, remaining_args = parser.parse_known_args()

    if args.interface == 'cli':
        # Pass remaining args to CLI
        cli_main(remaining_args)
    else:
        # Launch GUI
        gui_main()


if __name__ == "__main__":
    main()
