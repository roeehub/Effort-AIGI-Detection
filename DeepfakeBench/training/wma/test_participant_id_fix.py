#!/usr/bin/env python3
"""
Test for participant ID sanitization fix.

This test validates that corrupted participant IDs from gRPC messages
are properly sanitized before banner generation, ensuring per-participant
banners work correctly.
"""

import asyncio
import sys
import os

# Add the parent directory to sys.path to import the server module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import StreamingServiceImpl
import wma_streaming_pb2 as pb2


class MockParticipantFrame:
    """Mock ParticipantFrame for testing."""
    def __init__(self, participant_id):
        self.participant_id = participant_id


def test_participant_id_sanitization():
    """Test participant ID sanitization function."""
    print("=== Testing Participant ID Sanitization ===\n")
    
    service = StreamingServiceImpl()
    
    # Test cases: corrupted IDs that should be fixed
    test_cases = [
        # Clean IDs (should pass through with basic sanitization)
        ("participant_1", "participant_1"),
        ("participant_123", "participant_123"),
        
        # Corrupted IDs (the actual problem we're fixing)
        ('participant_1015",\n "fra', "participant_1015"),
        ('participant_202vf-400",\n  "frame_rate_hz"', "participant_202"),
        ('participant_5",\n    "total_participants"', "participant_5"),
        
        # Edge cases
        ('participant_999"chunk_id":"vf-1234"', "participant_999"),
        ('participant_42,\n"meeting_id":"test"', "participant_42"),
        ('participant_0{\n"corrupted":true}', "participant_0"),
        
        # Invalid/empty cases
        ("", "participant_unknown"),
        ('",\n "fra', "participant_unknown"),
        ("corrupted_data_no_participant", "corrupted_data_no_participant"),
    ]
    
    print("Testing participant ID sanitization:")
    all_passed = True
    
    for i, (input_id, expected_output) in enumerate(test_cases, 1):
        try:
            result = service._sanitize_participant_id(input_id)
            success = result == expected_output
            status = "PASS" if success else "FAIL"
            
            print(f"Test {i:2d}: {status}")
            print(f"  Input:    {repr(input_id)}")
            print(f"  Expected: {repr(expected_output)}")
            print(f"  Got:      {repr(result)}")
            print()
            
            if not success:
                all_passed = False
                
        except Exception as e:
            print(f"Test {i:2d}: EXCEPTION - {e}")
            print(f"  Input: {repr(input_id)}")
            print()
            all_passed = False
    
    return all_passed


async def test_banner_generation():
    """Test banner generation with sanitized participant IDs."""
    print("=== Testing Banner Generation with Sanitized IDs ===\n")
    
    service = StreamingServiceImpl()
    
    # Create mock participant frames with corrupted IDs
    corrupted_frames = [
        MockParticipantFrame('participant_1015",\n "fra'),
        MockParticipantFrame('participant_202vf-400",\n  "frame_rate_hz"'),
        MockParticipantFrame("participant_5"),  # Clean ID for comparison
    ]
    
    try:
        # This should not crash and should generate banners with clean IDs
        banners = await service._generate_per_person_banners(corrupted_frames)
        
        print(f"Generated {len(banners)} banner responses")
        
        # Check that banners were generated (even if probability is low)
        # The important thing is that it didn't crash
        print("PASS - Banner generation completed without errors")
        
        for i, banner_response in enumerate(banners):
            if hasattr(banner_response, 'screen_banner') and banner_response.screen_banner:
                banner = banner_response.screen_banner
                participant_id = getattr(banner, 'participant_id', 'N/A')
                print(f"  Banner {i+1}: participant_id = {repr(participant_id)}")
        
        print()
        return True
        
    except Exception as e:
        print(f"FAIL - Banner generation failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_end_to_end_fix():
    """Test the complete fix end-to-end."""
    print("=== End-to-End Fix Validation ===\n")
    
    print("Before Fix Simulation:")
    print("  Input: 'participant_1015\",\\n \"fra'")
    print("  Old Behavior: Sent corrupted ID to UI overlay -> treated as global banner")
    print()
    
    service = StreamingServiceImpl()
    corrupted_id = 'participant_1015",\n "fra'
    sanitized_id = service._sanitize_participant_id(corrupted_id)
    
    print("After Fix:")
    print(f"  Input: {repr(corrupted_id)}")
    print(f"  Sanitized: {repr(sanitized_id)}")
    print("  Result: Clean ID sent to UI overlay -> rendered as per-participant banner")
    print()
    
    # Verify the fix works
    if sanitized_id == "participant_1015":
        print("PASS - Fix working correctly!")
        return True
    else:
        print(f"FAIL - Expected 'participant_1015', got {repr(sanitized_id)}")
        return False


def main():
    """Run all tests."""
    print("Windows Meeting Assistant - Participant ID Fix Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("Participant ID Sanitization", test_participant_id_sanitization),
        ("Banner Generation", lambda: asyncio.run(test_banner_generation())),
        ("End-to-End Fix", test_end_to_end_fix),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((test_name, False))
        print("-" * 60)
    
    # Summary
    print("\nTest Summary:")
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! The participant ID fix is working correctly.")
        print("\nWhat this fix does:")
        print("- Sanitizes corrupted participant IDs in the Python backend")
        print("- Ensures per-participant banners use clean IDs") 
        print("- Prevents banners from being treated as global banners")
        print("- Enables correct PNG rendering on participant rectangles")
    else:
        print("Some tests failed. Please check the implementation.")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())