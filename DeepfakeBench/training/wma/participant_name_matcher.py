"""
Participant name matching utility using Levenshtein distance.
Handles OCR errors and minor variations in participant names.
"""

import re
import logging
from typing import Optional, Dict
import Levenshtein


class ParticipantNameMatcher:
    """
    Matches participant names using normalized Levenshtein distance to handle OCR errors.
    
    The similarity threshold controls how aggressive the matching is:
    - 0.0: Only exact matches
    - 0.2: Very conservative (catches obvious typos only)
    - 0.3: Moderate (recommended default - catches common OCR errors)
    - 0.4: Aggressive (may merge distinct participants)
    - 0.5+: Very aggressive (not recommended)
    """
    
    def __init__(self, similarity_threshold: float = 0.3):
        """
        Initialize the name matcher.
        
        Args:
            similarity_threshold: Maximum normalized distance to consider a match (0.0-1.0)
                                Lower = more strict, Higher = more lenient
        """
        self.similarity_threshold = similarity_threshold
        self.known_participants: Dict[str, str] = {}  # normalized_name -> original_name
        logging.info(f"[ParticipantNameMatcher] Initialized with threshold={similarity_threshold:.2f}")
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize a participant name for comparison.
        
        - Convert to lowercase
        - Remove extra whitespace
        - Remove common punctuation that might be OCR errors
        
        Args:
            name: Raw participant name
            
        Returns:
            Normalized name for comparison
        """
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove common OCR artifacts and punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Collapse multiple spaces into one
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate normalized Levenshtein distance between two names.
        
        Returns a value between 0.0 (identical) and 1.0 (completely different).
        
        Args:
            name1: First normalized name
            name2: Second normalized name
            
        Returns:
            Normalized distance (0.0 = identical, 1.0 = completely different)
        """
        if not name1 or not name2:
            return 1.0
        
        # Calculate Levenshtein distance
        distance = Levenshtein.distance(name1, name2)
        
        # Normalize by the length of the longer string
        max_len = max(len(name1), len(name2))
        if max_len == 0:
            return 0.0
        
        normalized_distance = distance / max_len
        return normalized_distance
    
    def find_matching_participant(self, raw_name: str) -> Optional[str]:
        """
        Find a matching participant name from known participants.
        
        Args:
            raw_name: The raw participant name to match
            
        Returns:
            The matched canonical participant name if found, None otherwise
        """
        if not raw_name or not raw_name.strip():
            return None
        
        # Normalize the input name
        normalized_input = self._normalize_name(raw_name)
        
        # Quick exact match check
        if normalized_input in self.known_participants:
            canonical_name = self.known_participants[normalized_input]
            if canonical_name != raw_name:
                logging.info(f"[ParticipantNameMatcher] Exact normalized match: '{raw_name}' -> '{canonical_name}'")
            return canonical_name
        
        # Find best match among known participants
        best_match = None
        best_distance = float('inf')
        
        for known_normalized, known_canonical in self.known_participants.items():
            distance = self._calculate_similarity(normalized_input, known_normalized)
            
            if distance < best_distance:
                best_distance = distance
                best_match = known_canonical
        
        # Check if best match is within threshold
        if best_match and best_distance <= self.similarity_threshold:
            similarity_pct = (1.0 - best_distance) * 100
            logging.info(f"[ParticipantNameMatcher] ✅ SIMILARITY MATCH FOUND!")
            logging.info(f"[ParticipantNameMatcher]   Input: '{raw_name}'")
            logging.info(f"[ParticipantNameMatcher]   Matched to: '{best_match}'")
            logging.info(f"[ParticipantNameMatcher]   Similarity: {similarity_pct:.1f}% (distance: {best_distance:.3f})")
            logging.info(f"[ParticipantNameMatcher]   Normalized input: '{normalized_input}'")
            logging.info(f"[ParticipantNameMatcher]   Normalized match: '{self._get_normalized_for_canonical(best_match)}'")
            return best_match
        
        return None
    
    def _get_normalized_for_canonical(self, canonical_name: str) -> str:
        """Get the normalized version of a canonical name."""
        for norm, canon in self.known_participants.items():
            if canon == canonical_name:
                return norm
        return self._normalize_name(canonical_name)
    
    def register_participant(self, canonical_name: str):
        """
        Register a new participant name as canonical.
        
        Args:
            canonical_name: The canonical form of the participant name
        """
        normalized = self._normalize_name(canonical_name)
        
        if normalized not in self.known_participants:
            self.known_participants[normalized] = canonical_name
            logging.info(f"[ParticipantNameMatcher] Registered new participant: '{canonical_name}' "
                  f"(normalized: '{normalized}')")
        else:
            existing = self.known_participants[normalized]
            if existing != canonical_name:
                logging.warning(f"[ParticipantNameMatcher] ⚠️ Normalized collision: '{canonical_name}' "
                      f"normalizes same as existing '{existing}'")
    
    def reset(self):
        """Clear all known participants."""
        count = len(self.known_participants)
        self.known_participants.clear()
        logging.info(f"[ParticipantNameMatcher] Reset: cleared {count} known participants")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about known participants."""
        return {
            "known_participants": len(self.known_participants),
            "similarity_threshold": self.similarity_threshold,
            "participant_names": list(self.known_participants.values())
        }
