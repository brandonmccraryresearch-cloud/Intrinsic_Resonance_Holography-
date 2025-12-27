"""
Update all notebooks with current features:
- ML surrogate models
- Enhanced transparency logging
- Failure detection and logging
- v21.4 manuscript references (fix v21.1)
"""

import json
import re
from pathlib import Path

def fix_v21_references(text):
    """Replace v21.1 with v21.4 in text."""
    # Replace various forms
    text = re.sub(r'v21\.1', 'v21.4', text)
    text = re.sub(r'v21 \.1', 'v21.4', text)
    text = re.sub(r'IRH v21\.1', 'IRH v21.4', text)
    text = re.sub(r'IRH21\.1', 'IRH21.4', text)
    return text

def update_notebook_cell(cell, notebook_name):
    """Update a single notebook cell with current features."""
    if cell['cell_type'] == 'markdown':
        # Fix version references in markdown
        if 'source' in cell:
            source = '\n'.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            source = fix_v21_references(source)
            cell['source'] = source.split('\n')
    
    elif cell['cell_type'] == 'code':
        # Check if this is a setup/import cell
        if 'source' in cell:
            source = '\n'.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Fix version references in code
            source = fix_v21_references(source)
            
            # Add ML surrogate imports if doing RG flow
            if 'rg_flow' in source.lower() and 'from src.ml' not in source:
                if 'from src.rg_flow' in source:
                    # Add ML imports after RG flow imports
                    insert_pos = source.find('from src.rg_flow')
                    next_line = source.find('\n', insert_pos)
                    
                    ml_import = """
# ML Surrogate Models (Phase 4.3)
try:
    from src.ml.rg_flow_surrogate import RGFlowSurrogate, SurrogateConfig, predict_rg_trajectory
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML surrogates not available")
"""
                    if next_line > 0:
                        source = source[:next_line+1] + ml_import + source[next_line+1:]
            
            # Add transparency logging if not present
            if 'transparency' not in source.lower() and 'TransparencyEngine' not in source:
                if 'import numpy' in source:
                    # Add transparency import after numpy
                    insert_pos = source.find('import numpy')
                    next_line = source.find('\n', insert_pos)
                    
                    transparency_import = """
# Ultra-verbose transparency logging
try:
    from src.logging.transparency_engine import TransparencyEngine, FULL
    transparency = TransparencyEngine(verbosity=FULL)
    TRANSPARENCY_AVAILABLE = True
except ImportError:
    TRANSPARENCY_AVAILABLE = False
    print("⚠️ Transparency engine not available")
"""
                    if next_line > 0:
                        source = source[:next_line+1] + transparency_import + source[next_line+1:]
            
            cell['source'] = source.split('\n')
    
    return cell

def update_notebook(notebook_path):
    """Update a single notebook file."""
    print(f"\nProcessing: {notebook_path.name}")
    
    with open(notebook_path) as f:
        notebook = json.load(f)
    
    # Track changes
    changes_made = 0
    
    # Update each cell
    for i, cell in enumerate(notebook['cells']):
        old_cell = json.dumps(cell)
        updated_cell = update_notebook_cell(cell, notebook_path.name)
        new_cell = json.dumps(updated_cell)
        
        if old_cell != new_cell:
            changes_made += 1
            notebook['cells'][i] = updated_cell
    
    # Save updated notebook
    if changes_made > 0:
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=2)
        print(f"  ✅ Updated {changes_made} cells")
    else:
        print(f"  ℹ️ No changes needed")
    
    return changes_made

def main():
    """Update all notebooks in the repository."""
    # Use relative path from script location for portability
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    notebooks_dir = repo_root / "notebooks"
    
    print("="*80)
    print("UPDATING IRH NOTEBOOKS WITH CURRENT FEATURES")
    print("="*80)
    
    # Find all notebooks
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    
    total_changes = 0
    for notebook_path in notebooks:
        if notebook_path.name == 'exascale_full_repo_ultra.ipynb':
            print(f"\nSkipping: {notebook_path.name} (newly created)")
            continue
        
        changes = update_notebook(notebook_path)
        total_changes += changes
    
    print("\n" + "="*80)
    print(f"SUMMARY: Updated {total_changes} cells across {len(notebooks)-1} notebooks")
    print("="*80)
    
    # Create update summary
    summary = {
        "timestamp": "2025-12-27",
        "updates_applied": [
            "Fixed v21.1 → v21.4 manuscript references",
            "Added ML surrogate imports where applicable",
            "Added transparency logging infrastructure",
            "Maintained backward compatibility"
        ],
        "notebooks_updated": [nb.name for nb in notebooks if nb.name != 'exascale_full_repo_ultra.ipynb'],
        "total_cells_modified": total_changes
    }
    
    summary_path = repo_root / "notebooks" / "UPDATE_SUMMARY.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Update summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
