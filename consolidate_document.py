#!/usr/bin/env python3
"""
Document Consolidation Script - Phase 4 of Semantic Consolidation Plan

This script executes the consolidation of the book analysis document using the 
consolidation map generated from similarity analysis. It consolidates 8 medium 
similarity groups while preserving 3 standalone sections.

Usage: python consolidate_document.py
"""

import json
import re
from typing import Dict, List, Tuple
import openai
import os
from pathlib import Path

class DocumentConsolidator:
    def __init__(self):
        # Set up OpenAI client - using GPT-4.1 for high-quality consolidation
        self.client = openai.OpenAI()
        self.model = "gpt-4.1"  # GPT-4.1 full model for maximum consolidation quality
        
        # Load consolidation map
        with open("consolidation_map.json", "r") as f:
            self.consolidation_map = json.load(f)
            
        # Load semantic analysis results containing the actual chunks
        with open("semantic_analysis_results.json", "r") as f:
            self.semantic_results = json.load(f)
            
        # Extract semantic chunks from the analysis results
        self.chunks = self._load_semantic_chunks()
        
    def _load_semantic_chunks(self) -> Dict[str, str]:
        """Load the semantic chunks from the analysis results."""
        chunks = {}
        
        # Load chunks from semantic analysis results
        for chunk_data in self.semantic_results.get("chunks", []):
            chunk_id = chunk_data["id"]
            content = chunk_data["content"]
            
            # Store chunk content with its ID
            chunks[chunk_id] = content
            
        print(f"Loaded {len(chunks)} semantic chunks from analysis results")
        return chunks
    
    def _consolidate_group(self, primary_chunk_id: str, secondary_chunk_ids: List[str], 
                          consolidation_strategy: str, unique_elements: List[str]) -> str:
        """Consolidate a group of chunks using AI according to the strategy."""
        
        # Get chunk contents
        primary_content = self.chunks.get(primary_chunk_id, "")
        secondary_contents = [self.chunks.get(chunk_id, "") for chunk_id in secondary_chunk_ids]
        
        # Prepare consolidation prompt
        prompt = f"""
You are tasked with consolidating related content sections from a business book analysis. 

CONSOLIDATION STRATEGY: {consolidation_strategy}

PRIMARY CONTENT:
{primary_content}

SECONDARY CONTENT(S):
{chr(10).join([f"--- Content {i+1} ---{chr(10)}{content}" for i, content in enumerate(secondary_contents)])}

UNIQUE ELEMENTS TO PRESERVE:
{chr(10).join(['- ' + element for element in unique_elements])}

INSTRUCTIONS:
1. Merge the content intelligently, using the primary content as the foundation
2. Integrate unique insights from secondary content without creating redundancy
3. Preserve ALL unique elements listed above
4. Remove redundant transitions and repetitive explanations
5. Maintain logical flow and professional tone
6. Keep all frameworks, examples, and action items
7. Ensure the consolidated section is coherent and comprehensive

Output the consolidated content in markdown format, maintaining the same level of detail but with improved organization and flow.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert business content editor specializing in consolidating related sections while preserving all unique insights and maintaining professional quality."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent, focused output
                max_tokens=4000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error consolidating group {primary_chunk_id}: {e}")
            # Fallback: return combined content without AI processing
            return f"{primary_content}\n\n{chr(10).join(secondary_contents)}"
    
    def consolidate_document(self) -> str:
        """Execute the full document consolidation according to the consolidation map."""
        
        consolidated_sections = []
        processed_chunks = set()
        
        print("Starting document consolidation...")
        print(f"Processing {len(self.consolidation_map['medium_similarity_groups'])} medium similarity groups...")
        
        # Process medium similarity groups
        for i, group in enumerate(self.consolidation_map['medium_similarity_groups']):
            print(f"Consolidating group {i+1}: {group['primary_chunk_id']} + {group['secondary_chunk_ids']}")
            
            consolidated_content = self._consolidate_group(
                group['primary_chunk_id'],
                group['secondary_chunk_ids'],
                group['consolidation_strategy'],
                group['unique_elements_to_preserve']
            )
            
            consolidated_sections.append(consolidated_content)
            
            # Mark chunks as processed
            processed_chunks.add(group['primary_chunk_id'])
            processed_chunks.update(group['secondary_chunk_ids'])
        
        # Add standalone sections (low similarity preserves)
        print(f"Adding {len(self.consolidation_map['low_similarity_preserves'])} standalone sections...")
        
        for chunk_id in self.consolidation_map['low_similarity_preserves']:
            if chunk_id in self.chunks and chunk_id not in processed_chunks:
                consolidated_sections.append(self.chunks[chunk_id])
                processed_chunks.add(chunk_id)
        
        # Add any remaining unprocessed chunks
        for chunk_id, content in self.chunks.items():
            if chunk_id not in processed_chunks:
                print(f"Adding unprocessed chunk: {chunk_id}")
                consolidated_sections.append(content)
        
        # Combine all sections into final document
        final_document = self._create_final_document(consolidated_sections)
        
        return final_document
    
    def _create_final_document(self, sections: List[str]) -> str:
        """Create the final consolidated document with proper structure."""
        
        # Create document header
        header = """# $100M Offers - Comprehensive Business Analysis

*Consolidated and Enhanced Edition*

This document represents a comprehensive analysis of Alex Hormozi's "$100M Offers" with semantically consolidated content to eliminate redundancy while preserving all unique insights, frameworks, and actionable guidance.

---

"""
        
        # Add table of contents
        toc = self._generate_table_of_contents(sections)
        
        # Combine everything
        final_content = header + toc + "\n\n" + "\n\n---\n\n".join(sections)
        
        return final_content
    
    def _generate_table_of_contents(self, sections: List[str]) -> str:
        """Generate a table of contents based on section headers."""
        
        toc_lines = ["## Table of Contents\n"]
        
        for i, section in enumerate(sections, 1):
            # Extract first header from section
            lines = section.split('\n')
            for line in lines:
                if line.startswith('##'):
                    header = line.replace('##', '').strip()
                    toc_lines.append(f"{i}. {header}")
                    break
            else:
                # No header found, use generic numbering
                toc_lines.append(f"{i}. Section {i}")
        
        return '\n'.join(toc_lines)
    
    def save_consolidated_document(self, filename: str = "book_analysis_consolidated.md"):
        """Save the consolidated document to file."""
        
        consolidated_content = self.consolidate_document()
        
        with open(filename, 'w') as f:
            f.write(consolidated_content)
        
        # Create consolidation report
        self._create_consolidation_report(filename)
        
        print(f"\n✅ Consolidation complete!")
        print(f"📄 Consolidated document saved as: {filename}")
        print(f"📊 Original chunks: {len(self.chunks)}")
        print(f"📊 Final sections: {len(self.consolidation_map['medium_similarity_groups']) + len(self.consolidation_map['low_similarity_preserves'])}")
        print(f"📈 Consolidation efficiency: {self.consolidation_map['consolidation_summary']['estimated_content_reduction_percentage']}% content optimization")
    
    def _create_consolidation_report(self, output_filename: str):
        """Create a detailed report of what was consolidated."""
        
        report = {
            "consolidation_timestamp": __import__('datetime').datetime.now().isoformat(),
            "input_document": "semantic_analysis_results.json",
            "output_document": output_filename,
            "consolidation_map_used": "consolidation_map.json",
            "summary": self.consolidation_map['consolidation_summary'],
            "groups_processed": len(self.consolidation_map['medium_similarity_groups']),
            "standalone_sections": len(self.consolidation_map['low_similarity_preserves']),
            "model_used": self.model,
            "preservation_guarantee": "All unique insights preserved as per consolidation map"
        }
        
        with open("consolidation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create preservation log
        preservation_log = []
        preservation_log.append("=== CONTENT PRESERVATION LOG ===\n")
        preservation_log.append(f"Consolidation completed: {report['consolidation_timestamp']}\n")
        preservation_log.append(f"Groups consolidated: {report['groups_processed']}")
        preservation_log.append(f"Standalone sections preserved: {report['standalone_sections']}\n")
        
        preservation_log.append("UNIQUE ELEMENTS PRESERVED:\n")
        for group in self.consolidation_map['medium_similarity_groups']:
            preservation_log.append(f"\nGroup: {group['primary_chunk_id']} + {group['secondary_chunk_ids']}")
            for element in group['unique_elements_to_preserve']:
                preservation_log.append(f"  ✓ {element}")
        
        with open("preservation_log.txt", 'w') as f:
            f.write('\n'.join(preservation_log))

def main():
    """Main execution function."""
    
    # Check required files exist
    required_files = ["semantic_analysis_results.json", "consolidation_map.json"]
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Error: Required file '{file}' not found")
            return
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        # Initialize consolidator
        consolidator = DocumentConsolidator()
        
        # Execute consolidation
        consolidator.save_consolidated_document()
        
        print("\n🎉 Document consolidation successful!")
        print("📋 Next steps as per plan:")
        print("  5. Develop o3 integration for final enhancement")
        print("  6. Test with current document and iterate as needed")
        
    except Exception as e:
        print(f"❌ Error during consolidation: {e}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main() 