"""
Update Manager for Experimental Data

THEORETICAL FOUNDATION: IRH21.md §7

This module coordinates updates from CODATA and PDG APIs, generates
comprehensive diff reports, and manages the update workflow.

Features:
- Unified interface for all experimental data updates
- Comprehensive diff reporting
- Update notification system
- Integration with CI/CD pipelines

Example:
    >>> from src.experimental.update_manager import UpdateManager
    >>> manager = UpdateManager()
    >>> report = manager.check_all_updates()
    >>> print(report)

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .codata_api import CODATAAPIClient, ConstantUpdate, check_codata_updates
from .pdg_api import PDGAPIClient, ParticleUpdate, check_pdg_updates

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §7"


@dataclass
class UpdateSummary:
    """
    Summary of all experimental data updates.
    
    Attributes
    ----------
    timestamp : datetime
        When the check was performed
    codata_updates : list of ConstantUpdate
        CODATA constant updates
    pdg_updates : list of ParticleUpdate
        PDG particle updates
    total_updates : int
        Total number of updates found
    significant_updates : int
        Number of significant updates (>3σ)
    requires_action : bool
        Whether any significant updates were found
    """
    timestamp: datetime
    codata_updates: List[ConstantUpdate]
    pdg_updates: List[ParticleUpdate]
    total_updates: int
    significant_updates: int
    requires_action: bool
    
    @classmethod
    # Theoretical Reference: IRH v21.4

    def from_updates(
        cls,
        codata_updates: List[ConstantUpdate],
        pdg_updates: List[ParticleUpdate],
    ) -> 'UpdateSummary':
        """Create summary from update lists."""
        total = len(codata_updates) + len(pdg_updates)
        
        significant_codata = sum(1 for u in codata_updates if u.is_significant)
        significant_pdg = sum(1 for u in pdg_updates if u.is_significant)
        significant = significant_codata + significant_pdg
        
        return cls(
            timestamp=datetime.now(),
            codata_updates=codata_updates,
            pdg_updates=pdg_updates,
            total_updates=total,
            significant_updates=significant,
            requires_action=significant > 0,
        )


class UpdateManager:
    """
    Manager for experimental data updates.
    
    THEORETICAL FOUNDATION: IRH21.md §7
    
    This class coordinates updates from multiple sources (CODATA, PDG)
    and provides a unified interface for checking updates, generating
    reports, and triggering notifications.
    
    Example:
        >>> manager = UpdateManager()
        >>> summary = manager.check_all_updates()
        >>> if summary.requires_action:
        ...     report = manager.generate_comprehensive_report(summary)
        ...     manager.notify(report)
    
    Notes
    -----
    The manager uses caching to minimize API calls. Cache TTL can be
    configured in the constructor.
    """
    
    # Theoretical Reference: IRH v21.4

    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_ttl: int = 86400,  # 24 hours
        notification_webhook: Optional[str] = None,
    ):
        """
        Initialize Update Manager.
        
        Parameters
        ----------
        cache_dir : Path, optional
            Base directory for caching (default: ~/.cache/irh)
        cache_ttl : int
            Cache time-to-live in seconds (default: 86400 = 24 hours)
        notification_webhook : str, optional
            Webhook URL for sending notifications (e.g., Slack, Discord)
        """
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'irh'
        self.cache_ttl = cache_ttl
        self.notification_webhook = notification_webhook
        
        # Initialize API clients
        self.codata_client = CODATAAPIClient(
            cache_dir=self.cache_dir / 'codata',
            cache_ttl=cache_ttl,
        )
        self.pdg_client = PDGAPIClient(
            cache_dir=self.cache_dir / 'pdg',
            cache_ttl=cache_ttl,
        )
    
    # Theoretical Reference: IRH v21.4

    
    def check_all_updates(
        self,
        force_refresh: bool = False,
    ) -> UpdateSummary:
        """
        Check for updates from all sources.
        
        Parameters
        ----------
        force_refresh : bool
            If True, bypass cache and force fresh API calls
        
        Returns
        -------
        UpdateSummary
            Summary of all updates found
        
        Examples
        --------
        >>> manager = UpdateManager()
        >>> summary = manager.check_all_updates()
        >>> print(f"Found {summary.total_updates} updates")
        >>> print(f"Significant: {summary.significant_updates}")
        """
        # If force refresh, clear cache
        if force_refresh:
            self._clear_cache()
        
        # Check CODATA updates
        codata_updates = check_codata_updates(cache_ttl=self.cache_ttl)
        
        # Check PDG updates
        pdg_updates = check_pdg_updates(cache_ttl=self.cache_ttl)
        
        # Create summary
        return UpdateSummary.from_updates(codata_updates, pdg_updates)
    
    # Theoretical Reference: IRH v21.4

    
    def generate_comprehensive_report(
        self,
        summary: UpdateSummary,
        format: str = 'markdown',
        include_insignificant: bool = True,
    ) -> str:
        """
        Generate comprehensive update report.
        
        Parameters
        ----------
        summary : UpdateSummary
            Update summary to report
        format : str
            Output format ('markdown', 'text', 'json', or 'html')
        include_insignificant : bool
            Whether to include updates below 3σ threshold
        
        Returns
        -------
        str
            Formatted comprehensive report
        
        Examples
        --------
        >>> manager = UpdateManager()
        >>> summary = manager.check_all_updates()
        >>> report = manager.generate_comprehensive_report(summary)
        >>> print(report)
        """
        if format == 'json':
            return self._generate_json_report(summary, include_insignificant)
        elif format == 'html':
            return self._generate_html_report(summary, include_insignificant)
        elif format == 'text':
            return self._generate_text_report(summary, include_insignificant)
        else:  # markdown
            return self._generate_markdown_report(summary, include_insignificant)
    
    def _generate_markdown_report(
        self,
        summary: UpdateSummary,
        include_insignificant: bool,
    ) -> str:
        """Generate Markdown report."""
        lines = [
            "# Experimental Data Update Report",
            "",
            f"**Generated**: {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Total Updates**: {summary.total_updates}",
            f"**Significant Updates**: {summary.significant_updates}",
            f"**Action Required**: {'⚠️ YES' if summary.requires_action else 'No'}",
            "",
        ]
        
        # CODATA section
        if summary.codata_updates:
            lines.append("## CODATA Fundamental Constants")
            lines.append("")
            lines.append("| Constant | Old Value | New Value | Δ (σ) | Status |")
            lines.append("|----------|-----------|-----------|-------|--------|")
            
            for update in summary.codata_updates:
                if not include_insignificant and not update.is_significant:
                    continue
                
                status = "⚠️ Significant" if update.is_significant else "Normal"
                lines.append(
                    f"| {update.name} | "
                    f"{update.old_value.value:.6e} ± {update.old_value.uncertainty:.1e} | "
                    f"{update.new_value.value:.6e} ± {update.new_value.uncertainty:.1e} | "
                    f"{update.change_sigma:.2f} | "
                    f"{status} |"
                )
            lines.append("")
        
        # PDG section
        if summary.pdg_updates:
            lines.append("## PDG Particle Data")
            lines.append("")
            lines.append("| Particle | Property | Old Value | New Value | Δ (σ) | Status |")
            lines.append("|----------|----------|-----------|-----------|-------|--------|")
            
            for update in summary.pdg_updates:
                if not include_insignificant and not update.is_significant:
                    continue
                
                status = "⚠️ Significant" if update.is_significant else "Normal"
                lines.append(
                    f"| {update.particle_name} | "
                    f"{update.property_name} | "
                    f"{update.old_value.value:.4e} ± {update.old_value.uncertainty:.1e} | "
                    f"{update.new_value.value:.4e} ± {update.new_value.uncertainty:.1e} | "
                    f"{update.change_sigma:.2f} | "
                    f"{status} |"
                )
            lines.append("")
        
        # Action items
        if summary.requires_action:
            lines.append("## Recommended Actions")
            lines.append("")
            lines.append("⚠️ **Significant updates detected (>3σ change)**")
            lines.append("")
            lines.append("1. Review all significant changes")
            lines.append("2. Update local database if changes are confirmed")
            lines.append("3. Re-run IRH predictions with updated values")
            lines.append("4. Update comparison tables in documentation")
            lines.append("5. Consider publishing an update note if predictions change significantly")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_text_report(
        self,
        summary: UpdateSummary,
        include_insignificant: bool,
    ) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 60,
            "EXPERIMENTAL DATA UPDATE REPORT",
            "=" * 60,
            "",
            f"Generated: {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Total Updates: {summary.total_updates}",
            f"Significant Updates: {summary.significant_updates}",
            f"Action Required: {'YES' if summary.requires_action else 'No'}",
            "",
        ]
        
        # CODATA section
        if summary.codata_updates:
            lines.append("-" * 60)
            lines.append("CODATA FUNDAMENTAL CONSTANTS")
            lines.append("-" * 60)
            lines.append("")
            
            for update in summary.codata_updates:
                if not include_insignificant and not update.is_significant:
                    continue
                
                lines.append(f"Constant: {update.name}")
                lines.append(f"  Old: {update.old_value.value:.9e} ± {update.old_value.uncertainty:.2e}")
                lines.append(f"  New: {update.new_value.value:.9e} ± {update.new_value.uncertainty:.2e}")
                lines.append(f"  Change: {update.change_sigma:.2f}σ")
                if update.is_significant:
                    lines.append("  ⚠️  SIGNIFICANT CHANGE")
                lines.append("")
        
        # PDG section
        if summary.pdg_updates:
            lines.append("-" * 60)
            lines.append("PDG PARTICLE DATA")
            lines.append("-" * 60)
            lines.append("")
            
            for update in summary.pdg_updates:
                if not include_insignificant and not update.is_significant:
                    continue
                
                lines.append(f"Particle: {update.particle_name} ({update.property_name})")
                lines.append(f"  Old: {update.old_value.value:.6e} ± {update.old_value.uncertainty:.2e} {update.old_value.unit}")
                lines.append(f"  New: {update.new_value.value:.6e} ± {update.new_value.uncertainty:.2e} {update.new_value.unit}")
                lines.append(f"  Change: {update.change_sigma:.2f}σ")
                if update.is_significant:
                    lines.append("  ⚠️  SIGNIFICANT CHANGE")
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_json_report(
        self,
        summary: UpdateSummary,
        include_insignificant: bool,
    ) -> str:
        """Generate JSON report."""
        data = {
            'timestamp': summary.timestamp.isoformat(),
            'total_updates': summary.total_updates,
            'significant_updates': summary.significant_updates,
            'requires_action': summary.requires_action,
            'codata_updates': [],
            'pdg_updates': [],
        }
        
        for update in summary.codata_updates:
            if not include_insignificant and not update.is_significant:
                continue
            
            data['codata_updates'].append({
                'name': update.name,
                'old_value': update.old_value.value,
                'old_uncertainty': update.old_value.uncertainty,
                'new_value': update.new_value.value,
                'new_uncertainty': update.new_value.uncertainty,
                'change_sigma': update.change_sigma,
                'is_significant': update.is_significant,
            })
        
        for update in summary.pdg_updates:
            if not include_insignificant and not update.is_significant:
                continue
            
            data['pdg_updates'].append({
                'particle': update.particle_name,
                'property': update.property_name,
                'old_value': update.old_value.value,
                'old_uncertainty': update.old_value.uncertainty,
                'new_value': update.new_value.value,
                'new_uncertainty': update.new_value.uncertainty,
                'unit': update.new_value.unit,
                'change_sigma': update.change_sigma,
                'is_significant': update.is_significant,
            })
        
        return json.dumps(data, indent=2)
    
    def _generate_html_report(
        self,
        summary: UpdateSummary,
        include_insignificant: bool,
    ) -> str:
        """Generate HTML report."""
        # Convert markdown to HTML (simple approach)
        md_report = self._generate_markdown_report(summary, include_insignificant)
        
        # Simple markdown to HTML conversion
        html_lines = ["<!DOCTYPE html>", "<html>", "<head>", 
                     "<meta charset='UTF-8'>",
                     "<title>IRH Update Report</title>",
                     "<style>",
                     "body { font-family: Arial, sans-serif; max-width: 1200px; margin: 20px auto; padding: 20px; }",
                     "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
                     "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
                     "th { background-color: #4CAF50; color: white; }",
                     ".significant { background-color: #ffebee; }",
                     "</style>",
                     "</head>", "<body>"]
        
        # Convert markdown lines to HTML
        for line in md_report.split('\n'):
            if line.startswith('# '):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith('## '):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith('| ') and '---' not in line:
                if '<table>' not in html_lines[-1]:
                    html_lines.append("<table>")
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if line.split('|')[1].strip() in ['Constant', 'Particle']:
                    html_lines.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
                else:
                    row_class = ' class="significant"' if '⚠️' in line else ''
                    html_lines.append(f"<tr{row_class}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            elif line.startswith('**'):
                html_lines.append(f"<p><strong>{line.replace('**', '')}</strong></p>")
            elif line.strip() and not line.startswith('|'):
                html_lines.append(f"<p>{line}</p>")
            elif '</tr>' in html_lines[-1] and not line.startswith('|'):
                html_lines.append("</table>")
        
        html_lines.extend(["</body>", "</html>"])
        return "\n".join(html_lines)
    
    def _clear_cache(self):
        """Clear all cached API responses."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Theoretical Reference: IRH v21.4

    
    def save_report(
        self,
        report: str,
        filename: Optional[Path] = None,
        format: str = 'markdown',
    ) -> Path:
        """
        Save report to file.
        
        Parameters
        ----------
        report : str
            Report content
        filename : Path, optional
            Output filename (auto-generated if not provided)
        format : str
            Report format for extension
        
        Returns
        -------
        Path
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ext = {'markdown': 'md', 'text': 'txt', 'json': 'json', 'html': 'html'}[format]
            filename = Path(f"update_report_{timestamp}.{ext}")
        
        filename = Path(filename)
        filename.write_text(report)
        
        return filename.absolute()
    
    # Theoretical Reference: IRH v21.4

    
    def notify(
        self,
        report: str,
        title: str = "IRH Experimental Data Update",
    ) -> bool:
        """
        Send notification via configured webhook.
        
        Parameters
        ----------
        report : str
            Report content to send
        title : str
            Notification title
        
        Returns
        -------
        bool
            True if notification was sent successfully
        
        Notes
        -----
        Requires notification_webhook to be configured.
        """
        if not self.notification_webhook:
            return False
        
        try:
            import urllib.request
            import json
            
            # Format for Slack/Discord webhook
            payload = {
                'text': title,
                'blocks': [
                    {
                        'type': 'section',
                        'text': {
                            'type': 'mrkdwn',
                            'text': report[:3000],  # Limit length
                        }
                    }
                ]
            }
            
            req = urllib.request.Request(
                self.notification_webhook,
                data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json'},
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        
        except Exception:
            return False


# Convenience function for CLI use
# Theoretical Reference: IRH v21.4

def generate_update_report(
    output_file: Optional[Path] = None,
    format: str = 'markdown',
    force_refresh: bool = False,
) -> Tuple[str, Path]:
    """
    Generate and save update report.
    
    Parameters
    ----------
    output_file : Path, optional
        Output file path (auto-generated if not provided)
    format : str
        Output format ('markdown', 'text', 'json', or 'html')
    force_refresh : bool
        Force fresh API calls, bypassing cache
    
    Returns
    -------
    report : str
        Report content
    filepath : Path
        Path to saved report file
    
    Examples
    --------
    >>> report, path = generate_update_report(format='markdown')
    >>> print(f"Report saved to {path}")
    """
    manager = UpdateManager()
    summary = manager.check_all_updates(force_refresh=force_refresh)
    report = manager.generate_comprehensive_report(summary, format=format)
    filepath = manager.save_report(report, output_file, format=format)
    
    return report, filepath
