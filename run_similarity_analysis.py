#!/usr/bin/env python3
"""
Run similarity analysis on existing semantic analysis results.
This implements step 1.3 of the semantic consolidation plan.
"""

import json
import os
from pathlib import Path
from analyze_semantic_content import SemanticAnalyzer, ContentChunk, SimilarityAnalysis
from typing import List, Dict, Any

def load_existing_analysis(file_path: str) -> Dict[str, Any]:
    """Load existing semantic analysis results."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading existing analysis: {e}")
        return {}

def convert_to_chunks(analysis_data: Dict[str, Any]) -> List[ContentChunk]:
    """Convert analysis data back to ContentChunk objects."""
    chunks = []
    for chunk_data in analysis_data.get('chunks', []):
        chunk = ContentChunk(
            id=chunk_data['id'],
            content=chunk_data['content'],
            start_line=chunk_data['start_line'],
            end_line=chunk_data['end_line'],
            topic_theme=chunk_data.get('topic_theme', ''),
            frameworks=chunk_data.get('frameworks', []),
            examples=chunk_data.get('examples', []),
            action_items=chunk_data.get('action_items', []),
            unique_insights=chunk_data.get('unique_insights', [])
        )
        chunks.append(chunk)
    return chunks

def print_similarity_summary(similarities: List[SimilarityAnalysis]):
    """Print a summary of similarity analysis results."""
    # Categorize similarities
    high_sim = [s for s in similarities if s.similarity_category == "high"]
    medium_sim = [s for s in similarities if s.similarity_category == "medium"]
    low_sim = [s for s in similarities if s.similarity_category == "low"]
    
    print("\n" + "="*80)
    print("SIMILARITY ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Total chunk pairs analyzed: {len(similarities)}")
    print(f"High similarity pairs (>80%): {len(high_sim)}")
    print(f"Medium similarity pairs (50-80%): {len(medium_sim)}")
    print(f"Low similarity pairs (<50%): {len(low_sim)}")
    
    # Print high similarity pairs
    if high_sim:
        print("\n" + "-"*60)
        print("HIGH SIMILARITY PAIRS (>80%)")
        print("-"*60)
        for sim in high_sim:
            print(f"\n{sim.chunk1_id} ↔ {sim.chunk2_id}")
            print(f"  Similarity: {sim.similarity_score}% | Overlap: {sim.overlap_percentage}%")
            print(f"  Overlap: {sim.overlap_description}")
            print(f"  Unique to {sim.chunk1_id}: {sim.unique_value_chunk1}")
            print(f"  Unique to {sim.chunk2_id}: {sim.unique_value_chunk2}")
    
    # Print medium similarity pairs
    if medium_sim:
        print("\n" + "-"*60)
        print("MEDIUM SIMILARITY PAIRS (50-80%)")
        print("-"*60)
        for sim in medium_sim:
            print(f"\n{sim.chunk1_id} ↔ {sim.chunk2_id}")
            print(f"  Similarity: {sim.similarity_score}% | Overlap: {sim.overlap_percentage}%")
            print(f"  Overlap: {sim.overlap_description}")
            print(f"  Unique to {sim.chunk1_id}: {sim.unique_value_chunk1}")
            print(f"  Unique to {sim.chunk2_id}: {sim.unique_value_chunk2}")
    
    print("\n" + "="*80)

def save_similarity_results(analysis_data: Dict[str, Any], similarities: List[SimilarityAnalysis], output_file: str):
    """Save the updated analysis results with similarity data."""
    # Convert similarities to dictionaries for JSON serialization
    similarities_dict = []
    for sim in similarities:
        sim_dict = {
            'chunk1_id': sim.chunk1_id,
            'chunk2_id': sim.chunk2_id,
            'similarity_score': sim.similarity_score,
            'similarity_category': sim.similarity_category,
            'overlap_percentage': sim.overlap_percentage,
            'unique_value_chunk1': sim.unique_value_chunk1,
            'unique_value_chunk2': sim.unique_value_chunk2,
            'overlap_description': sim.overlap_description
        }
        similarities_dict.append(sim_dict)
    
    # Update the analysis data with similarity results
    analysis_data['similarities'] = similarities_dict
    analysis_data['similarity_analysis_completed'] = True
    
    # Save updated results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated analysis results saved to: {output_file}")

def main():
    """Main function to run similarity analysis."""
    
    # Check if analysis results exist
    analysis_file = "semantic_analysis_results.json"
    if not os.path.exists(analysis_file):
        print(f"Error: {analysis_file} not found. Please run analyze_semantic_content.py first.")
        return
    
    print("Loading existing semantic analysis results...")
    analysis_data = load_existing_analysis(analysis_file)
    
    if not analysis_data:
        print("Failed to load analysis data.")
        return
    
    print(f"Loaded {len(analysis_data.get('chunks', []))} chunks from previous analysis.")
    
    # Convert to ContentChunk objects
    chunks = convert_to_chunks(analysis_data)
    
    # Check if similarity analysis already exists
    if 'similarities' in analysis_data and analysis_data.get('similarity_analysis_completed', False):
        print("Similarity analysis already completed. Skipping...")
        # Still print summary for existing results
        existing_similarities = []
        for sim_data in analysis_data['similarities']:
            sim = SimilarityAnalysis(
                chunk1_id=sim_data['chunk1_id'],
                chunk2_id=sim_data['chunk2_id'],
                similarity_score=sim_data['similarity_score'],
                similarity_category=sim_data['similarity_category'],
                overlap_percentage=sim_data['overlap_percentage'],
                unique_value_chunk1=sim_data['unique_value_chunk1'],
                unique_value_chunk2=sim_data['unique_value_chunk2'],
                overlap_description=sim_data['overlap_description']
            )
            existing_similarities.append(sim)
        print_similarity_summary(existing_similarities)
        return
    
    # Initialize analyzer
    print("Initializing semantic analyzer...")
    analyzer = SemanticAnalyzer()
    
    # Run similarity analysis
    print("Running similarity analysis...")
    similarities = analyzer.analyze_all_similarities(chunks)
    
    # Print summary
    print_similarity_summary(similarities)
    
    # Save results
    save_similarity_results(analysis_data, similarities, analysis_file)
    
    print("\nSimilarity analysis completed successfully!")
    print("Ready to proceed to Phase 2: Consolidation Strategy")

if __name__ == "__main__":
    main() 