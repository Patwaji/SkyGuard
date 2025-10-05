"""
Test script to verify all scripts can find their target directories
"""
import os
import sys

def test_script_paths():
    """Test that scripts can find data and model directories"""
    
    # Get paths relative to scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Define target directories
    directories = {
        'Raw Data': os.path.join(project_root, 'data', 'raw'),
        'Processed Data': os.path.join(project_root, 'data', 'processed'), 
        'Models': os.path.join(project_root, 'models'),
        'App Config': os.path.join(project_root, 'app', 'config'),
        'App Components': os.path.join(project_root, 'app', 'components')
    }
    
    print("🔍 Testing script path resolution...")
    print(f"📂 Script Directory: {script_dir}")
    print(f"📁 Project Root: {project_root}")
    print()
    
    # Test each directory
    all_good = True
    for name, path in directories.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_good = False
    
    print()
    if all_good:
        print("🎉 All directories found! Scripts should work correctly.")
    else:
        print("⚠️ Some directories missing. Please check project structure.")
    
    # Test environment config access
    try:
        sys.path.append(os.path.join(project_root, 'app', 'config'))
        from env_config import config
        print(f"✅ Environment config accessible")
        print(f"📍 Target City: {config.TARGET_CITY}")
    except ImportError as e:
        print(f"❌ Environment config not accessible: {e}")
    
    return all_good

if __name__ == '__main__':
    test_script_paths()
