"""Script to find and show dataclass issues"""

import re
from pathlib import Path

def find_dataclass_issues():
    """Find dataclasses with mutable defaults"""
    backend_dir = Path(__file__).parent
    
    for py_file in backend_dir.glob("*.py"):
        if py_file.name.startswith("test_"):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for @dataclass decorator
            if '@dataclass' in content:
                print(f"\nFound @dataclass in: {py_file.name}")
                
                # Look for common mutable defaults
                issues = []
                
                # Check for np.array([])
                if re.search(r':\s*np\.ndarray\s*=\s*np\.array\(\[', content):
                    issues.append("  - np.ndarray with np.array([]) default")
                
                # Check for list []
                if re.search(r':\s*list\s*=\s*\[\]', content):
                    issues.append("  - list with [] default")
                
                # Check for dict {}
                if re.search(r':\s*dict\s*=\s*\{\}', content):
                    issues.append("  - dict with {} default")
                
                if issues:
                    print("  Issues found:")
                    for issue in issues:
                        print(issue)
                    print(f"\n  Fix by using: field(default_factory=...)")
                    
        except Exception as e:
            print(f"Error reading {py_file.name}: {e}")

if __name__ == "__main__":
    print("Scanning for dataclass issues...\n")
    find_dataclass_issues()
    print("\n\nTo fix, change:")
    print("  AHN_data: np.ndarray = np.array([])")
    print("to:")
    print("  AHN_data: np.ndarray = field(default_factory=lambda: np.array([]))")
