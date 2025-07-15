#!/usr/bin/env python3
"""
Create consolidation mapping logic based on similarity analysis results.
This implements step 3 of Phase 2 in the semantic consolidation plan.
"""

import json
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class ConsolidationRule:
    """Represents a consolidation rule for merging similar chunks."""
    primary_chunk_id: str
    secondary_chunk_ids: List[str]
    consolidation_type: str  # "merge_high", "group_medium", "preserve_low"
    similarity_scores: List[float]
    consolidation_strategy: str
    unique_elements_to_preserve: List[str]

@dataclass
class ConsolidationMap:
    """Complete consolidation strategy for the document."""
    high_similarity_merges: List[ConsolidationRule]
    medium_similarity_groups: List[ConsolidationRule]
    low_similarity_preserves: List[str]
    total_chunks_original: int
    estimated_chunks_final: int
    consolidation_summary: Dict[str, Any]

def load_similarity_results(file_path: str) -> Dict[str, Any]:
    """Load similarity analysis results."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def categorize_similarities(similarities: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Categorize similarities into high, medium, and low groups."""
    high_sim = [s for s in similarities if s['similarity_category'] == 'high']
    medium_sim = [s for s in similarities if s['similarity_category'] == 'medium']
    low_sim = [s for s in similarities if s['similarity_category'] == 'low']
    
    return high_sim, medium_sim, low_sim

def create_merge_clusters(similarities: List[Dict]) -> List[List[str]]:
    """Create clusters of chunks that should be merged together."""
    clusters = []
    processed_chunks = set()
    
    # Build adjacency map
    adjacency = {}
    for sim in similarities:
        chunk1, chunk2 = sim['chunk1_id'], sim['chunk2_id']
        
        if chunk1 not in adjacency:
            adjacency[chunk1] = []
        if chunk2 not in adjacency:
            adjacency[chunk2] = []
            
        adjacency[chunk1].append(chunk2)
        adjacency[chunk2].append(chunk1)
    
    # Find connected components
    for chunk_id in adjacency:
        if chunk_id not in processed_chunks:
            cluster = []
            stack = [chunk_id]
            
            while stack:
                current = stack.pop()
                if current not in processed_chunks:
                    processed_chunks.add(current)
                    cluster.append(current)
                    stack.extend([neighbor for neighbor in adjacency[current] 
                                if neighbor not in processed_chunks])
            
            if len(cluster) > 1:
                clusters.append(sorted(cluster))
    
    return clusters

def select_primary_chunk(cluster: List[str], chunks_data: List[Dict]) -> str:
    """Select the primary chunk from a cluster based on content richness."""
    chunk_scores = {}
    
    for chunk_id in cluster:
        chunk_data = next(c for c in chunks_data if c['id'] == chunk_id)
        
        # Score based on content richness
        score = 0
        score += len(chunk_data.get('frameworks', [])) * 3
        score += len(chunk_data.get('examples', [])) * 2
        score += len(chunk_data.get('action_items', [])) * 2
        score += len(chunk_data.get('unique_insights', [])) * 3
        score += len(chunk_data.get('content', '').split()) * 0.1  # Word count with lower weight
        
        chunk_scores[chunk_id] = score
    
    return max(chunk_scores.items(), key=lambda x: x[1])[0]

def extract_unique_elements(cluster: List[str], chunks_data: List[Dict], similarities: List[Dict]) -> List[str]:
    """Extract unique elements that must be preserved when merging."""
    unique_elements = []
    
    for chunk_id in cluster:
        # Find similarity entries for this chunk
        for sim in similarities:
            if sim['chunk1_id'] == chunk_id:
                unique_elements.append(f"From {chunk_id}: {sim['unique_value_chunk1']}")
            elif sim['chunk2_id'] == chunk_id:
                unique_elements.append(f"From {chunk_id}: {sim['unique_value_chunk2']}")
    
    return unique_elements

def create_consolidation_rules(similarities: List[Dict], chunks_data: List[Dict], 
                             consolidation_type: str) -> List[ConsolidationRule]:
    """Create consolidation rules for a given similarity category."""
    if consolidation_type == "merge_high":
        # High similarity chunks should be merged using clustering
        clusters = create_merge_clusters(similarities)
        rules = []
        
        for cluster in clusters:
            if len(cluster) > 1:
                primary = select_primary_chunk(cluster, chunks_data)
                secondary = [c for c in cluster if c != primary]
                scores = []
                
                # Get similarity scores
                for sim in similarities:
                    if ((sim['chunk1_id'] in cluster and sim['chunk2_id'] in cluster) and
                        (sim['chunk1_id'] == primary or sim['chunk2_id'] == primary)):
                        scores.append(sim['similarity_score'])
                
                unique_elements = extract_unique_elements(cluster, chunks_data, similarities)
                
                rule = ConsolidationRule(
                    primary_chunk_id=primary,
                    secondary_chunk_ids=secondary,
                    consolidation_type=consolidation_type,
                    similarity_scores=scores,
                    consolidation_strategy="Merge all content into primary chunk, preserving unique insights from secondary chunks",
                    unique_elements_to_preserve=unique_elements
                )
                rules.append(rule)
        
        return rules
    
    elif consolidation_type == "group_medium":
        # Medium similarity: create smaller thematic groups, not one giant cluster
        rules = []
        
        # Sort similarities by score descending to prioritize stronger connections
        sorted_sims = sorted(similarities, key=lambda x: x['similarity_score'], reverse=True)
        processed_chunks = set()
        
        for sim in sorted_sims:
            chunk1, chunk2 = sim['chunk1_id'], sim['chunk2_id']
            
            # Skip if both chunks are already processed
            if chunk1 in processed_chunks and chunk2 in processed_chunks:
                continue
                
            # Create a group with these two chunks
            if chunk1 not in processed_chunks and chunk2 not in processed_chunks:
                # New group
                primary = select_primary_chunk([chunk1, chunk2], chunks_data)
                secondary = [chunk2 if primary == chunk1 else chunk1]
                
                unique_elements = extract_unique_elements([chunk1, chunk2], chunks_data, [sim])
                
                rule = ConsolidationRule(
                    primary_chunk_id=primary,
                    secondary_chunk_ids=secondary,
                    consolidation_type=consolidation_type,
                    similarity_scores=[sim['similarity_score']],
                    consolidation_strategy="Group related chunks with logical progression, remove redundant transitions",
                    unique_elements_to_preserve=unique_elements
                )
                rules.append(rule)
                processed_chunks.add(chunk1)
                processed_chunks.add(chunk2)
        
        return rules
    
    return []

def generate_consolidation_summary(original_chunks: int, rules: List[ConsolidationRule]) -> Dict[str, Any]:
    """Generate a summary of the consolidation plan."""
    chunks_to_merge = sum(len(rule.secondary_chunk_ids) + 1 for rule in rules if rule.consolidation_type == "merge_high")
    chunks_to_group = sum(len(rule.secondary_chunk_ids) + 1 for rule in rules if rule.consolidation_type == "group_medium")
    
    high_merges = [r for r in rules if r.consolidation_type == "merge_high"]
    medium_groups = [r for r in rules if r.consolidation_type == "group_medium"]
    
    estimated_reduction = sum(len(rule.secondary_chunk_ids) for rule in high_merges)
    estimated_final = original_chunks - estimated_reduction
    
    return {
        "original_chunks": original_chunks,
        "chunks_affected_by_high_similarity_merges": chunks_to_merge,
        "chunks_affected_by_medium_similarity_grouping": chunks_to_group,
        "high_similarity_merge_operations": len(high_merges),
        "medium_similarity_group_operations": len(medium_groups),
        "estimated_chunks_after_consolidation": estimated_final,
        "estimated_content_reduction_percentage": round((estimated_reduction / original_chunks) * 100, 1)
    }

def create_consolidation_map(analysis_data: Dict[str, Any]) -> ConsolidationMap:
    """Create a complete consolidation map from similarity analysis results."""
    similarities = analysis_data['similarities']
    chunks_data = analysis_data['chunks']
    
    high_sim, medium_sim, low_sim = categorize_similarities(similarities)
    
    print(f"Categorizing similarities:")
    print(f"  High similarity pairs (>80%): {len(high_sim)}")
    print(f"  Medium similarity pairs (50-80%): {len(medium_sim)}")
    print(f"  Low similarity pairs (<50%): {len(low_sim)}")
    
    # Create consolidation rules
    high_rules = create_consolidation_rules(high_sim, chunks_data, "merge_high")
    medium_rules = create_consolidation_rules(medium_sim, chunks_data, "group_medium")
    
    # Find chunks that appear in low similarity only (should be preserved as-is)
    all_chunk_ids = set(chunk['id'] for chunk in chunks_data)
    high_affected = set()
    medium_affected = set()
    
    for rule in high_rules:
        high_affected.add(rule.primary_chunk_id)
        high_affected.update(rule.secondary_chunk_ids)
    
    for rule in medium_rules:
        medium_affected.add(rule.primary_chunk_id)
        medium_affected.update(rule.secondary_chunk_ids)
    
    low_preserve = list(all_chunk_ids - high_affected - medium_affected)
    
    # Generate summary
    original_chunks = len(chunks_data)
    all_rules = high_rules + medium_rules
    summary = generate_consolidation_summary(original_chunks, all_rules)
    
    estimated_final = summary["estimated_chunks_after_consolidation"]
    
    return ConsolidationMap(
        high_similarity_merges=high_rules,
        medium_similarity_groups=medium_rules,
        low_similarity_preserves=low_preserve,
        total_chunks_original=original_chunks,
        estimated_chunks_final=estimated_final,
        consolidation_summary=summary
    )

def save_consolidation_map(consolidation_map: ConsolidationMap, output_file: str):
    """Save the consolidation map to a JSON file."""
    
    # Convert to dictionary for JSON serialization
    data = {
        "high_similarity_merges": [
            {
                "primary_chunk_id": rule.primary_chunk_id,
                "secondary_chunk_ids": rule.secondary_chunk_ids,
                "consolidation_type": rule.consolidation_type,
                "similarity_scores": rule.similarity_scores,
                "consolidation_strategy": rule.consolidation_strategy,
                "unique_elements_to_preserve": rule.unique_elements_to_preserve
            }
            for rule in consolidation_map.high_similarity_merges
        ],
        "medium_similarity_groups": [
            {
                "primary_chunk_id": rule.primary_chunk_id,
                "secondary_chunk_ids": rule.secondary_chunk_ids,
                "consolidation_type": rule.consolidation_type,
                "similarity_scores": rule.similarity_scores,
                "consolidation_strategy": rule.consolidation_strategy,
                "unique_elements_to_preserve": rule.unique_elements_to_preserve
            }
            for rule in consolidation_map.medium_similarity_groups
        ],
        "low_similarity_preserves": consolidation_map.low_similarity_preserves,
        "total_chunks_original": consolidation_map.total_chunks_original,
        "estimated_chunks_final": consolidation_map.estimated_chunks_final,
        "consolidation_summary": consolidation_map.consolidation_summary
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def print_consolidation_plan(consolidation_map: ConsolidationMap):
    """Print a detailed consolidation plan."""
    print("\n" + "="*80)
    print("CONSOLIDATION MAP & STRATEGY")
    print("="*80)
    
    print(f"\nOVERVIEW:")
    print(f"  Original chunks: {consolidation_map.total_chunks_original}")
    print(f"  Estimated final chunks: {consolidation_map.estimated_chunks_final}")
    print(f"  Reduction: {consolidation_map.consolidation_summary['estimated_content_reduction_percentage']}%")
    
    print(f"\nHIGH SIMILARITY MERGES ({len(consolidation_map.high_similarity_merges)} operations):")
    for i, rule in enumerate(consolidation_map.high_similarity_merges, 1):
        print(f"  {i}. Merge into {rule.primary_chunk_id}:")
        print(f"     Secondary chunks: {', '.join(rule.secondary_chunk_ids)}")
        print(f"     Similarity scores: {rule.similarity_scores}")
        print(f"     Strategy: {rule.consolidation_strategy}")
        print()
    
    print(f"MEDIUM SIMILARITY GROUPS ({len(consolidation_map.medium_similarity_groups)} operations):")
    for i, rule in enumerate(consolidation_map.medium_similarity_groups, 1):
        print(f"  {i}. Group around {rule.primary_chunk_id}:")
        print(f"     Related chunks: {', '.join(rule.secondary_chunk_ids)}")
        print(f"     Similarity scores: {rule.similarity_scores}")
        print(f"     Strategy: {rule.consolidation_strategy}")
        print()
    
    print(f"LOW SIMILARITY PRESERVES ({len(consolidation_map.low_similarity_preserves)} chunks):")
    print(f"  Preserve as distinct sections: {', '.join(consolidation_map.low_similarity_preserves)}")
    
    print("\n" + "="*80)

def main():
    """Main function to create consolidation mapping."""
    
    # Load similarity analysis results
    analysis_file = "semantic_analysis_results.json"
    if not os.path.exists(analysis_file):
        print(f"Error: {analysis_file} not found.")
        return
    
    print("Loading similarity analysis results...")
    analysis_data = load_similarity_results(analysis_file)
    
    # Create consolidation map
    print("Creating consolidation map...")
    consolidation_map = create_consolidation_map(analysis_data)
    
    # Print the plan
    print_consolidation_plan(consolidation_map)
    
    # Save consolidation map
    output_file = "consolidation_map.json"
    save_consolidation_map(consolidation_map, output_file)
    print(f"\nConsolidation map saved to: {output_file}")
    
    print("\nNext steps:")
    print("1. Review the consolidation plan above")
    print("2. Execute document consolidation using this map")
    print("3. Proceed to o3 integration for final enhancement")

if __name__ == "__main__":
    main() 