#!/usr/bin/env python3
"""
Fix figure saving in notebooks by finding cells with plt.show() and adding save code.
"""

import json
import os
from pathlib import Path

NOTEBOOK_CONFIGS = {
    'basic_tabular': {
        'image_dir': 'images/tutorials/basic_tabular',
        'figures': [
            {'pattern': 'Original Data.*2D Projection', 'filename': 'original_data_2d.png'},
            {'pattern': 'Plot spatial coverage', 'filename': 'spatial_coverage_comparison.png'},
            {'pattern': 'Plot marginal distributions', 'filename': 'marginal_distributions.png'},
            {'pattern': 'Bar chart comparing metrics', 'filename': 'metrics_comparison.png'},
        ]
    }
}

def find_cell_by_pattern(notebook, pattern):
    """Find cell index by pattern in source or markdown before it."""
    for i, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if pattern.lower() in source.lower():
            # Check if next cell has plt.show()
            if i + 1 < len(notebook['cells']):
                next_cell = notebook['cells'][i + 1]
                if next_cell['cell_type'] == 'code':
                    next_source = ''.join(next_cell.get('source', []))
                    if 'plt.show()' in next_source:
                        return i + 1
            # Or check current cell
            if 'plt.show()' in source:
                return i
    return None

def add_save_code(cell, image_dir, filename):
    """Add save code before plt.show()."""
    source_lines = cell.get('source', [])
    if isinstance(source_lines, list):
        source = ''.join(source_lines)
    else:
        source = source_lines
    
    if 'plt.show()' not in source:
        return False
    
    if 'plt.savefig' in source:
        return False  # Already has save code
    
    # Create save code
    save_code = f"""# Save figure
import os
os.makedirs('{image_dir}', exist_ok=True)
plt.savefig('{image_dir}/{filename}', dpi=150, bbox_inches='tight')
"""
    
    # Replace plt.show() with save code + plt.show()
    new_source = source.replace('plt.show()', save_code + '\nplt.show()')
    
    # Convert back to list format
    cell['source'] = new_source.split('\n')
    # Add newline characters
    cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line 
                      for i, line in enumerate(cell['source'])]
    
    return True

def process_notebook(notebook_name, config):
    """Process notebook."""
    notebook_path = Path(f'tutorials/{notebook_name}.ipynb')
    
    if not notebook_path.exists():
        print(f"⚠️  Not found: {notebook_path}")
        return False
    
    print(f"\n📝 Processing {notebook_name}.ipynb...")
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    image_dir = config['image_dir']
    os.makedirs(image_dir, exist_ok=True)
    
    for fig_config in config['figures']:
        pattern = fig_config['pattern']
        filename = fig_config['filename']
        
        cell_idx = find_cell_by_pattern(notebook, pattern)
        if cell_idx is None:
            print(f"  ⚠️  Could not find cell with pattern: {pattern}")
            continue
        
        cell = notebook['cells'][cell_idx]
        if add_save_code(cell, image_dir, filename):
            print(f"  ✓ Added save code: {filename}")
        else:
            print(f"  ⚠️  Could not add save code: {filename}")
    
    # Save
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"  ✅ Saved")
    return True

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    for name, config in NOTEBOOK_CONFIGS.items():
        process_notebook(name, config)

