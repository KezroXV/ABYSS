import pandas as pd
import os
import json

# Liste des keywords qui indiquent un "fix commit"
FIX_KEYWORDS = [
    'fix', 'bug', 'issue', 'error', 'crash', 'fail',
    'patch', 'hotfix', 'correct', 'resolve', 'closes',
    'repair', 'broken', 'defect', 'problem', 'revert'
]

def is_fix_commit(message):
    message = message.lower()
    
    for keyword in FIX_KEYWORDS:
      if keyword in message:
        return True
      else:
        return False
    return False

# Tests à la fin du fichier
if __name__ == "__main__":
    # Test 1: Fix commit
    assert is_fix_commit("fix: authentication bug") == True
    print("✓ Test 1 passed")
    
    # Test 2: Fix avec majuscule
    assert is_fix_commit("Fix crash on startup") == True
    print("✓ Test 2 passed")
    
    # Test 3: Pas un fix
    assert is_fix_commit("add new login page") == False
    print("✓ Test 3 passed")
    
    # Test 4: Hotfix
    assert is_fix_commit("hotfix: security issue") == True
    print("✓ Test 4 passed")
    
    print("🎉 All tests passed!")