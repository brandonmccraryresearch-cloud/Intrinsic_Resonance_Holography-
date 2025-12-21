#!/usr/bin/env python
"""
CLI tool for checking experimental data updates.

THEORETICAL FOUNDATION: IRH21.md §7

This script provides a command-line interface for checking CODATA and PDG
updates, generating reports, and managing the update workflow.

Usage:
    python scripts/check_experimental_updates.py [options]

Examples:
    # Check for updates (uses cache)
    python scripts/check_experimental_updates.py
    
    # Force fresh API calls
    python scripts/check_experimental_updates.py --force-refresh
    
    # Generate HTML report
    python scripts/check_experimental_updates.py --format html --output report.html
    
    # Only show significant updates
    python scripts/check_experimental_updates.py --significant-only

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experimental.update_manager import UpdateManager


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Check for experimental data updates (CODATA, PDG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                  # Check with default settings
  %(prog)s --force-refresh                  # Force fresh API calls
  %(prog)s --format html -o report.html    # Generate HTML report
  %(prog)s --significant-only               # Only show significant updates
  %(prog)s --notify WEBHOOK_URL             # Send notification
        """,
    )
    
    parser.add_argument(
        '--format',
        choices=['markdown', 'text', 'json', 'html'],
        default='markdown',
        help='Output format (default: markdown)',
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output file path (default: auto-generated)',
    )
    
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force fresh API calls, bypass cache',
    )
    
    parser.add_argument(
        '--significant-only',
        action='store_true',
        help='Only show updates with >3σ change',
    )
    
    parser.add_argument(
        '--cache-ttl',
        type=int,
        default=86400,
        help='Cache time-to-live in seconds (default: 86400 = 24h)',
    )
    
    parser.add_argument(
        '--notify',
        type=str,
        metavar='WEBHOOK_URL',
        help='Send notification to webhook URL (Slack/Discord)',
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output',
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='IRH Experimental Data Update Checker v21.0',
    )
    
    args = parser.parse_args()
    
    # Create update manager
    if not args.quiet:
        print("IRH Experimental Data Update Checker v21.0")
        print("=" * 60)
        print()
    
    manager = UpdateManager(
        cache_ttl=args.cache_ttl,
        notification_webhook=args.notify,
    )
    
    # Check for updates
    if not args.quiet:
        print("Checking for experimental data updates...")
        if args.force_refresh:
            print("(Force refresh - bypassing cache)")
        print()
    
    summary = manager.check_all_updates(force_refresh=args.force_refresh)
    
    # Print summary
    if not args.quiet:
        print(f"Total updates found: {summary.total_updates}")
        print(f"Significant updates (>3σ): {summary.significant_updates}")
        print(f"Action required: {'⚠️ YES' if summary.requires_action else 'No'}")
        print()
    
    # Generate report
    report = manager.generate_comprehensive_report(
        summary,
        format=args.format,
        include_insignificant=not args.significant_only,
    )
    
    # Save report
    output_file = manager.save_report(report, args.output, args.format)
    
    if not args.quiet:
        print(f"Report saved to: {output_file}")
        print()
        
        # Print report to console if text or markdown
        if args.format in ['text', 'markdown'] and not args.significant_only:
            print("=" * 60)
            print(report)
        
        # Show action items if needed
        if summary.requires_action:
            print()
            print("⚠️ " + "=" * 58)
            print("⚠️  SIGNIFICANT UPDATES DETECTED - ACTION REQUIRED")
            print("⚠️ " + "=" * 58)
            print()
            print("Recommended actions:")
            print("  1. Review all significant changes (>3σ)")
            print("  2. Verify updates are legitimate (not data errors)")
            print("  3. Update local database if changes confirmed")
            print("  4. Re-run IRH predictions with updated values")
            print("  5. Update comparison tables in documentation")
            print()
    
    # Send notification if requested and significant updates found
    if args.notify and summary.requires_action:
        if not args.quiet:
            print("Sending notification...")
        
        success = manager.notify(
            report,
            title=f"⚠️ IRH Experimental Data Update Alert ({summary.significant_updates} significant)"
        )
        
        if not args.quiet:
            if success:
                print("✓ Notification sent successfully")
            else:
                print("✗ Failed to send notification")
    
    # Exit code
    if summary.requires_action:
        sys.exit(1)  # Exit with error code if action required
    else:
        sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
