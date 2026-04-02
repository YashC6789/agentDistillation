"""
Validation script for golden trajectories.

This script checks the quality and format of golden trajectories before SFT training.
Run this locally or in Colab before starting training to catch issues early.

Usage:
    python validate_trajectories.py --input gsm8k_golden_trajectories_new.jsonl
"""

import json
import argparse
import re
from collections import Counter
from typing import Dict, List, Any


def load_trajectories(filepath: str) -> List[Dict[str, Any]]:
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                trajectories.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
    return trajectories


def validate_format(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate the format of trajectories."""
    issues = []
    valid_count = 0
    
    for i, traj in enumerate(trajectories):
        # Check for 'messages' field
        if 'messages' not in traj:
            issues.append(f"Sample {i}: Missing 'messages' field")
            continue
        
        messages = traj['messages']
        
        # Check if messages is a list
        if not isinstance(messages, list):
            issues.append(f"Sample {i}: 'messages' is not a list")
            continue
        
        # Check if there are at least 2 messages (user + assistant)
        if len(messages) < 2:
            issues.append(f"Sample {i}: Less than 2 messages")
            continue
        
        # Validate message structure
        for j, msg in enumerate(messages):
            if 'role' not in msg:
                issues.append(f"Sample {i}, Message {j}: Missing 'role'")
            if 'content' not in msg:
                issues.append(f"Sample {i}, Message {j}: Missing 'content'")
            if msg.get('role') not in ['user', 'assistant', 'system']:
                issues.append(f"Sample {i}, Message {j}: Invalid role '{msg.get('role')}'")
        
        # Check conversation structure
        if messages[0].get('role') != 'user':
            issues.append(f"Sample {i}: First message should be from 'user'")
        if messages[-1].get('role') != 'assistant':
            issues.append(f"Sample {i}: Last message should be from 'assistant'")
        
        if not issues:
            valid_count += 1
    
    return {
        'valid_count': valid_count,
        'total_count': len(trajectories),
        'issues': issues[:20]  # Show first 20 issues
    }


def analyze_content(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the content of trajectories."""
    lengths = []
    answer_formats = []
    has_thinking = 0
    
    for traj in trajectories:
        if 'messages' not in traj:
            continue
        
        messages = traj['messages']
        
        # Get assistant response
        assistant_msgs = [m for m in messages if m.get('role') == 'assistant']
        if assistant_msgs:
            response = assistant_msgs[-1].get('content', '')
            
            # Check length
            lengths.append(len(response))
            
            # Check for common answer patterns
            if re.search(r'[Tt]he answer is:?\s*\d+', response):
                answer_formats.append('answer_is')
            elif re.search(r'\\boxed\{\d+\}', response):
                answer_formats.append('boxed')
            else:
                answer_formats.append('other')
            
            # Check for thinking/reasoning
            if 'step by step' in response.lower() or 'let\'s think' in response.lower():
                has_thinking += 1
    
    return {
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
        'answer_formats': Counter(answer_formats),
        'thinking_count': has_thinking,
        'thinking_percentage': (has_thinking / len(trajectories) * 100) if trajectories else 0
    }


def check_duplicates(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check for duplicate questions."""
    questions = []
    
    for traj in trajectories:
        if 'messages' not in traj:
            continue
        messages = traj['messages']
        user_msgs = [m for m in messages if m.get('role') == 'user']
        if user_msgs:
            questions.append(user_msgs[0].get('content', ''))
    
    question_counts = Counter(questions)
    duplicates = {q: count for q, count in question_counts.items() if count > 1}
    
    return {
        'unique_questions': len(question_counts),
        'total_questions': len(questions),
        'duplicate_count': len(duplicates),
        'duplicates': list(duplicates.items())[:5]  # Show first 5
    }


def estimate_training_time(trajectories: List[Dict[str, Any]], model_size: str = "0.5B") -> None:
    """Estimate training time based on dataset size."""
    num_samples = len(trajectories)
    
    # Rough estimates (in minutes) per 100 samples for A100
    time_per_100 = {
        "0.5B": 15,
        "1.5B": 30,
        "7B": 60
    }
    
    base_time = time_per_100.get(model_size, 15)
    estimated_minutes = (num_samples / 100) * base_time
    estimated_hours = estimated_minutes / 60
    
    print(f"\n⏱️  Estimated Training Time ({model_size} model on A100):")
    print(f"   Dataset size: {num_samples} samples")
    print(f"   Estimated time: ~{estimated_hours:.1f} hours ({estimated_minutes:.0f} minutes)")
    print(f"   Note: Add ~20-30 minutes for setup and saving")


def main():
    parser = argparse.ArgumentParser(description='Validate golden trajectories for SFT')
    parser.add_argument('--input', type=str, required=True, help='Path to JSONL file')
    parser.add_argument('--model-size', type=str, default='0.5B', 
                        choices=['0.5B', '1.5B', '7B'], 
                        help='Model size for time estimation')
    args = parser.parse_args()
    
    print("="*60)
    print("GOLDEN TRAJECTORIES VALIDATION")
    print("="*60)
    print(f"Input file: {args.input}\n")
    
    # Load trajectories
    print("Loading trajectories...")
    trajectories = load_trajectories(args.input)
    print(f"Loaded {len(trajectories)} trajectories\n")
    
    if len(trajectories) == 0:
        print("❌ Error: No trajectories loaded!")
        return
    
    # 1. Format validation
    print("1️⃣  Validating format...")
    format_results = validate_format(trajectories)
    print(f"   Valid samples: {format_results['valid_count']}/{format_results['total_count']}")
    
    if format_results['issues']:
        print(f"   ⚠️  Found {len(format_results['issues'])} issues:")
        for issue in format_results['issues'][:5]:
            print(f"      - {issue}")
        if len(format_results['issues']) > 5:
            print(f"      ... and {len(format_results['issues']) - 5} more")
    else:
        print("   ✅ All samples have valid format!")
    
    # 2. Content analysis
    print("\n2️⃣  Analyzing content...")
    content_results = analyze_content(trajectories)
    print(f"   Average response length: {content_results['avg_length']:.0f} characters")
    print(f"   Length range: {content_results['min_length']} - {content_results['max_length']}")
    print(f"   Responses with reasoning: {content_results['thinking_percentage']:.1f}%")
    print(f"   Answer formats: {dict(content_results['answer_formats'])}")
    
    # 3. Check for duplicates
    print("\n3️⃣  Checking for duplicates...")
    dup_results = check_duplicates(trajectories)
    print(f"   Unique questions: {dup_results['unique_questions']}")
    print(f"   Duplicate questions: {dup_results['duplicate_count']}")
    
    if dup_results['duplicates']:
        print(f"   ⚠️  Found duplicates:")
        for q, count in dup_results['duplicates'][:3]:
            print(f"      - \"{q[:60]}...\" appears {count} times")
    else:
        print("   ✅ No duplicate questions found!")
    
    # 4. Show sample
    print("\n4️⃣  Sample trajectory:")
    print("-" * 60)
    if trajectories and 'messages' in trajectories[0]:
        sample = trajectories[0]['messages']
        for msg in sample:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')[:200]
            print(f"\n{role}:")
            print(f"{content}...")
    print("-" * 60)
    
    # 5. Training time estimate
    estimate_training_time(trajectories, args.model_size)
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_valid = (
        format_results['valid_count'] == format_results['total_count'] and
        content_results['thinking_percentage'] > 50 and
        dup_results['duplicate_count'] == 0
    )
    
    if all_valid:
        print("✅ Dataset looks good! Ready for training.")
    else:
        print("⚠️  Dataset has some issues:")
        if format_results['valid_count'] < format_results['total_count']:
            print("   - Format validation errors detected")
        if content_results['thinking_percentage'] < 50:
            print("   - Many responses lack reasoning")
        if dup_results['duplicate_count'] > 0:
            print("   - Duplicate questions found")
        print("\nConsider fixing these issues before training.")
    
    print("="*60)


if __name__ == "__main__":
    main()
