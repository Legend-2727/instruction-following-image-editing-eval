#!/usr/bin/env python3
"""
test_judge_parsing.py — Lightweight smoke tests for strict VLM judge parsing.

Tests that the parser correctly handles valid JSON, invalid JSON, and
normalizes labels according to canonical rules.

Run: cd scripts && python test_judge_parsing.py
"""

import sys
from pathlib import Path

# Add parent to path to import utils
sys.path.insert(0, str(Path(__file__).parent))

from utils.vlm_evaluator import QwenVLMJudge
from utils.schema import ERROR_TO_IDX

def test_valid_json_success():
    """Test: Valid JSON with Success adherence."""
    raw = '{"adherence": "Success", "error_types": [], "confidence": 0.95}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert result["adherence"] == "Success"
    assert result["error_types"] == []
    assert result["confidence"] == 0.95
    print("✓ test_valid_json_success")

def test_valid_json_partial():
    """Test: Valid JSON with Partial and 2 error types (canonical order)."""
    raw = '{"adherence": "Partial", "error_types": ["Wrong Object", "Missing Object"], "confidence": 0.72}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert result["adherence"] == "Partial"
    # Canonical order: Missing Object (idx 1) < Wrong Object (idx 0)
    assert len(result["error_types"]) == 2
    assert set(result["error_types"]) == {"Missing Object", "Wrong Object"}
    assert result["confidence"] == 0.72
    print("✓ test_valid_json_partial")

def test_no_with_empty_taxonomy_becomes_under_editing():
    """Test: No adherence + empty errors -> should add Under-editing."""
    raw = '{"adherence": "No", "error_types": [], "confidence": 0.6}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert result["adherence"] == "No"
    assert result["error_types"] == ["Under-editing"]
    print("✓ test_no_with_empty_taxonomy_becomes_under_editing")

def test_success_forces_empty_taxonomy():
    """Test: Success + non-empty errors -> force empty."""
    raw = '{"adherence": "Success", "error_types": ["Wrong Object"], "confidence": 0.99}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert result["adherence"] == "Success"
    assert result["error_types"] == []
    print("✓ test_success_forces_empty_taxonomy")

def test_deduplication():
    """Test: Duplicate error types are deduplicated."""
    raw = '{"adherence": "Partial", "error_types": ["Wrong Object", "Wrong Object", "Missing Object"], "confidence": 0.7}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert len(result["error_types"]) == 2
    assert set(result["error_types"]) == {"Missing Object", "Wrong Object"}
    print("✓ test_deduplication")

def test_invalid_adherence():
    """Test: Invalid adherence value -> parse_failed=True."""
    raw = '{"adherence": "Maybe", "error_types": [], "confidence": 0.7}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is True
    assert result["adherence"] is None
    print("✓ test_invalid_adherence")

def test_invalid_error_type():
    """Test: Invalid error type -> parse_failed=True, no fuzzy matching."""
    raw = '{"adherence": "Partial", "error_types": ["Wrong Color"], "confidence": 0.7}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is True
    assert result["adherence"] is None
    print("✓ test_invalid_error_type")

def test_invalid_confidence_range():
    """Test: Confidence out of range -> parse_failed=True."""
    raw = '{"adherence": "Partial", "error_types": [], "confidence": 1.5}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is True
    print("✓ test_invalid_confidence_range")

def test_bad_json():
    """Test: Malformed JSON -> parse_failed=True."""
    raw = '{"adherence": "Partial", error_types: }'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is True
    assert result["adherence"] is None
    print("✓ test_bad_json")

def test_markdown_fences_stripped():
    """Test: Markdown code fences are stripped."""
    raw = '```json\n{"adherence": "Partial", "error_types": [], "confidence": 0.8}\n```'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert result["adherence"] == "Partial"
    print("✓ test_markdown_fences_stripped")

def test_cap_at_2_error_types():
    """Test: More than 2 error types get capped."""
    raw = '{"adherence": "Partial", "error_types": ["Wrong Object", "Missing Object", "Extra Object"], "confidence": 0.7}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert len(result["error_types"]) == 2
    print("✓ test_cap_at_2_error_types")

def main():
    """Run all tests."""
    print("Running judge parsing smoke tests...\n")
    
    test_valid_json_success()
    test_valid_json_partial()
    test_no_with_empty_taxonomy_becomes_under_editing()
    test_success_forces_empty_taxonomy()
    test_deduplication()
    test_invalid_adherence()
    test_invalid_error_type()
    test_invalid_confidence_range()
    test_bad_json()
    test_markdown_fences_stripped()
    test_cap_at_2_error_types()
    
    print("\n✅ All smoke tests passed!")

if __name__ == "__main__":
    main()

def test_no_with_empty_taxonomy_becomes_under_editing():
    """Test: No adherence + empty errors -> should add Under-editing."""
    raw = '{"adherence": "No", "error_types": [], "confidence": 0.6}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert result["adherence"] == "No"
    assert result["error_types"] == ["Under-editing"]
    print("✓ test_no_with_empty_taxonomy_becomes_under_editing")

def test_success_forces_empty_taxonomy():
    """Test: Success + non-empty errors -> force empty."""
    raw = '{"adherence": "Success", "error_types": ["Wrong Object"], "confidence": 0.99}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert result["adherence"] == "Success"
    assert result["error_types"] == []
    print("✓ test_success_forces_empty_taxonomy")

def test_deduplication():
    """Test: Duplicate error types are deduplicated (canonical order)."""
    raw = '{"adherence": "Partial", "error_types": ["Wrong Object", "Wrong Object", "Missing Object"], "confidence": 0.7}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    # Should have 2 unique types in canonical order
    assert len(result["error_types"]) == 2
    assert set(result["error_types"]) == {"Missing Object", "Wrong Object"}
    # Verify canonical ordering (Wrong Object idx=0, Missing Object idx=1)
    assert result["error_types"] == ["Wrong Object", "Missing Object"]
    print("✓ test_deduplication")

def test_invalid_adherence():
    """Test: Invalid adherence value -> parse_failed=True."""
    raw = '{"adherence": "Maybe", "error_types": [], "confidence": 0.7}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is True
    assert result["adherence"] is None
    print("✓ test_invalid_adherence")

def test_invalid_error_type():
    """Test: Invalid error type -> parse_failed=True, no fuzzy matching."""
    raw = '{"adherence": "Partial", "error_types": ["Wrong Color"], "confidence": 0.7}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is True
    assert result["adherence"] is None
    print("✓ test_invalid_error_type")

def test_invalid_confidence_range():
    """Test: Confidence out of range -> parse_failed=True."""
    raw = '{"adherence": "Partial", "error_types": [], "confidence": 1.5}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is True
    print("✓ test_invalid_confidence_range")

def test_bad_json():
    """Test: Malformed JSON -> parse_failed=True."""
    raw = '{"adherence": "Partial", error_types: }'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is True
    assert result["adherence"] is None
    print("✓ test_bad_json")

def test_markdown_fences_stripped():
    """Test: Markdown code fences are stripped."""
    raw = '```json\n{"adherence": "Partial", "error_types": [], "confidence": 0.8}\n```'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert result["adherence"] == "Partial"
    print("✓ test_markdown_fences_stripped")

def test_cap_at_2_error_types():
    """Test: More than 2 error types get capped."""
    raw = '{"adherence": "Partial", "error_types": ["Wrong Object", "Missing Object", "Extra Object"], "confidence": 0.7}'
    result = QwenVLMJudge._parse_response(raw)
    assert result["parse_failed"] is False
    assert len(result["error_types"]) == 2
    print("✓ test_cap_at_2_error_types")

def main():
    """Run all tests."""
    print("Running judge parsing smoke tests...\n")
    
    test_valid_json_success()
    test_valid_json_partial()
    test_no_with_empty_taxonomy_becomes_under_editing()
    test_success_forces_empty_taxonomy()
    test_deduplication()
    test_invalid_adherence()
    test_invalid_error_type()
    test_invalid_confidence_range()
    test_bad_json()
    test_markdown_fences_stripped()
    test_cap_at_2_error_types()
    
    print("\n✅ All smoke tests passed!")

if __name__ == "__main__":
    main()
