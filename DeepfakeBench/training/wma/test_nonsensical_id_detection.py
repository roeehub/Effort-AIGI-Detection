#!/usr/bin/env python3
"""
Test script to validate the nonsensical participant ID detection functionality.
"""

import sys
import os

# Add the wma directory to path so we can import the server module
sys.path.insert(0, os.path.dirname(__file__))

# Import the server module to access the StreamingServiceImpl class
from server import StreamingServiceImpl

def test_nonsensical_id_detection():
    """Test the _is_nonsensical_id method with various test cases."""
    
    # Create a service instance to test the method
    service = StreamingServiceImpl()
    
    # Test cases: (input_id, expected_result, description)
    test_cases = [
        # Valid participant IDs (should NOT be flagged as nonsensical)
        ("participant_123", False, "Normal participant ID"),
        ("participant_1234abc", False, "Participant ID with alphanumeric"),
        ("participant-456", False, "Participant ID with hyphen"),
        ("participant_test_1", False, "Participant ID with underscores"),
        ("participant.789", False, "Participant ID with dot"),
        ("user_123", False, "Short valid ID"),
        ("", False, "Empty string (handled by isEmpty check, not nonsensical)"),
        ("  ", False, "Whitespace only (handled by isEmpty check)"),
        
        # Nonsensical IDs (should be flagged as nonsensical)
        ("participant_123$$$###", True, "ID with multiple special chars at end"),
        ("$%#@participant_123", True, "ID with special chars at start"),
        ("part$&*#icipant_123", True, "ID with special chars in middle"),
        ("!!!###$$$%%%", True, "Only special characters"),
        ("participant_123$%^&*()@#", True, "ID with >30% special chars"),
        ("user@#$%^&*()!~`", True, "Short ID with many specials"),
        ("participant_123&&&&####", True, "ID with consecutive special chars"),
        ("abc$#@%^&*()!~def", True, "Mixed with high special char ratio"),
        ("participant_123$$$", True, "ID with 3+ consecutive specials"),
        ("a$b$c$d$e$f$g$h$i$j$k", True, "ID with >10 special characters total"),
        
        # Edge cases
        ("participant_123$", False, "ID with single special char (should pass)"),
        ("participant_123$$", False, "ID with two consecutive specials (should pass)"),
        ("participant_123!@", False, "ID with two non-consecutive specials (should pass)"),
        ("participant_123!@#$%^&*()", True, "ID with exactly 10 special chars"),
        ("participan_123!@#$%^&*()!", True, "ID with 11 special chars (threshold exceeded)"),
    ]
    
    print("🧪 Testing Nonsensical Participant ID Detection")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for input_id, expected_nonsensical, description in test_cases:
        try:
            result = service._is_nonsensical_id(input_id)
            
            if result == expected_nonsensical:
                status = "✅ PASS"
                passed += 1
            else:
                status = "❌ FAIL"
                failed += 1
            
            print(f"{status} | {description}")
            print(f"      Input: '{input_id}'")
            print(f"      Expected: {expected_nonsensical}, Got: {result}")
            print()
            
        except Exception as e:
            print(f"❌ ERROR | {description}")
            print(f"        Input: '{input_id}'")
            print(f"        Exception: {e}")
            print()
            failed += 1
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"💥 {failed} tests failed!")
        return False


def test_sanitize_participant_id_with_nonsensical():
    """Test the complete _sanitize_participant_id method including nonsensical ID rejection."""
    
    # Create a service instance to test the method
    service = StreamingServiceImpl()
    
    print("\n🧪 Testing Complete _sanitize_participant_id Method")
    print("=" * 60)
    
    # Test cases: (input_id, expected_result_type, description)
    test_cases = [
        # Valid IDs (should return sanitized string)
        ("participant_123", str, "Normal participant ID"),
        ("participant_456_test", str, "Valid ID with underscores"),
        
        # Nonsensical IDs (should return None)
        ("participant_123$$$###", type(None), "ID with many special chars"),
        ("$%#@participant_123", type(None), "ID with special chars at start"),
        ("!!!###$$$%%%", type(None), "Only special characters"),
        ("participant_123!@#$%^&*()", type(None), "ID with >10 special chars"),
        
        # JSON corruption (should return sanitized string if not nonsensical)
        ('participant_1015vf-400",\n  "fra', str, "JSON corrupted ID (should extract participant_1015vf-400)"),
        ("participant_123\"corrupted", str, "ID with JSON quotes"),
    ]
    
    passed = 0
    failed = 0
    
    for input_id, expected_type, description in test_cases:
        try:
            result = service._sanitize_participant_id(input_id)
            
            if type(result) == expected_type:
                status = "✅ PASS"
                passed += 1
                if result is not None:
                    print(f"{status} | {description}")
                    print(f"      Input: '{input_id}' -> Output: '{result}'")
                else:
                    print(f"{status} | {description}")
                    print(f"      Input: '{input_id}' -> REJECTED (None)")
            else:
                status = "❌ FAIL"
                failed += 1
                print(f"{status} | {description}")
                print(f"      Input: '{input_id}'")
                print(f"      Expected type: {expected_type.__name__}, Got type: {type(result).__name__}")
                print(f"      Result: {result}")
            print()
            
        except Exception as e:
            print(f"❌ ERROR | {description}")
            print(f"        Input: '{input_id}'")
            print(f"        Exception: {e}")
            print()
            failed += 1
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"💥 {failed} tests failed!")
        return False


if __name__ == "__main__":
    print("🔍 Nonsensical Participant ID Detection Test Suite")
    print("This tests the ability to detect and reject participant IDs with too many special characters.")
    print()
    
    # Run the tests
    test1_passed = test_nonsensical_id_detection()
    test2_passed = test_sanitize_participant_id_with_nonsensical()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🚀 All test suites passed! Nonsensical ID detection is working correctly.")
        sys.exit(0)
    else:
        print("🚨 Some tests failed. Please review the implementation.")
        sys.exit(1)