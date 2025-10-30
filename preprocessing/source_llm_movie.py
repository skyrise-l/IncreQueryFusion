import os
import pandas as pd
import numpy as np
from utils.result_judge import ResultJudge
import json
import re
import time

def extract_source_name(filename):
    """Extract the first part of filename as source name"""
    base_name = os.path.splitext(filename)[0]
    source_name = re.split(r'[_\d]', base_name)[0]
    return source_name

def analyze_data_quality(file_path):
    """Analyze data quality of a single file"""
    try:
        df = pd.read_csv(file_path, sep='\t', dtype=str)
        
        if 'Source' in df.columns:
            df = df.drop('Source', axis=1)
        
        # Movie-related key columns
        columns_to_analyze = ['Title', 'Director']
        
        filename = os.path.basename(file_path)
        source_name = extract_source_name(filename)
        
        quality_info = {
            'file_name': filename,
            'source_name': source_name,
            'total_records': len(df),
            'columns_analysis': {},
            'sample_data': {}
        }
        
        for col in columns_to_analyze:
            if col not in df.columns:
                quality_info['columns_analysis'][col] = {
                    'exists': False,
                    'nan_ratio': 1.0,
                    'non_nan_count': 0
                }
                continue
                
            # Fix null value handling
            nan_mask = df[col].isna()
            # Handle string-type null values
            if df[col].dtype == 'object':
                nan_mask = nan_mask | (df[col] == '') | (df[col].str.lower() == 'nan')
            
            nan_ratio = nan_mask.mean()
            non_nan_count = len(df) - nan_mask.sum()
            
            quality_info['columns_analysis'][col] = {
                'exists': True,
                'nan_ratio': round(nan_ratio, 4),
                'non_nan_count': int(non_nan_count),
                'is_all_nan': nan_ratio == 1.0
            }
            
            non_nan_values = df[col][~nan_mask].head(3).tolist()
            quality_info['sample_data'][col] = non_nan_values
        
        return quality_info
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return None

def generate_batch_llm_prompt(batch_quality_infos, batch_size=5):
    """Generate LLM prompt for a batch of data sources"""
    
    batch_summary = []
    
    for i, quality_info in enumerate(batch_quality_infos):
        source_name = quality_info['source_name']
        
        source_summary = f"\n--- Data Source {i+1}: {source_name} ---\n"
        source_summary += f"Total records: {quality_info['total_records']}\n"
        
        # Movie-related key columns
        key_columns = ['title', 'director', 'year', 'genre']
        missing_info = []
        
        for col in key_columns:
            if col in quality_info['columns_analysis']:
                analysis = quality_info['columns_analysis'][col]
                if analysis['exists'] and not analysis['is_all_nan']:
                    missing_info.append(f"{col}: {analysis['nan_ratio']*100:.1f}% missing")
        
        source_summary += "Missing data: " + "; ".join(missing_info) + "\n"
        
        # Add sample data information
        if 'sample_data' in quality_info and quality_info['sample_data']:
            source_summary += "Sample data: "
            sample_info = []
            for col, samples in quality_info['sample_data'].items():
                if samples:
                    sample_info.append(f"{col} sample: {samples[0][:50]}..." if len(str(samples[0])) > 50 else f"{col} sample: {samples[0]}")
            source_summary += " | ".join(sample_info[:2]) + "\n"
        
        batch_summary.append(source_summary)
    
    example_sources = [info['source_name'] for info in batch_quality_infos]
    example_json = "{" + ", ".join([f'"{source}": level' for source in example_sources]) + "}"
    
    prompt = f"""
## Background
This is a data quality assessment task for multiple movie data sources. The data sources contain movie information such as title, director, year, genre, cast, rating, and box office information.

## Data Sources to Evaluate
{''.join(batch_summary)}

## Evaluation Requirements
Based on the data quality information and your domain knowledge about movie data sources (e.g., some well-known movie databases, streaming platforms, and film institutions are more trustworthy), please provide an initial confidence level for each data source (only the numerical level, no reasoning needed):
- 1: Highest confidence (excellent data quality, most trustworthy)
- 2: High confidence (good data quality, quite trustworthy)  
- 3: Medium confidence (average data quality, moderately trustworthy)
- 4: Low confidence (poor data quality, less trustworthy)
- 5: Lowest confidence (very poor data quality, least trustworthy)

Please return only a JSON dictionary in the following format, without any additional content:
{example_json}

Please ensure to use the source names as keys (not the full filenames).

Evaluation logic to consider:
1. First, analyze the source name to determine which authoritative institution it comes from (e.g., official movie databases, reputable streaming platforms, film archives, studios). This accounts for 60% of the evaluation.
2. Missing ratios of key attributes (title, director, year, genre) account for 25% of the evaluation.
3. Data volume (very small datasets reduce reliability) accounts for 15% of the evaluation.

Examples of authoritative movie data sources:
- Official movie databases: IMDb, TMDb, The Movie Database, etc.
- Streaming platforms: Netflix, Amazon Prime, Hulu, Disney+, etc.
- Film institutions: Academy Awards, Cannes Film Festival, Sundance, etc.
- Studios: Warner Bros, Universal, Paramount, Sony Pictures, etc.
- Box office tracking: Box Office Mojo, The Numbers, etc.
- Critical review sites: Rotten Tomatoes, Metacritic, etc.
"""
    return prompt

def save_results_to_file(results, output_file='movie_data_source_confidence.json'):
    """Save results to file"""
    simplified_results = []
    for result in results:
        simplified = {
            'file_name': result['file_name'],
            'source_name': result['source_name'],
            'confidence_level': result['confidence_level'],
            'total_records': result['quality_info']['total_records'],
            'key_columns_missing_ratio': {}
        }
        
        # Movie-related key columns
        key_columns = ['Title', 'Director']
        for col in key_columns:
            if col in result['quality_info']['columns_analysis']:
                analysis = result['quality_info']['columns_analysis'][col]
                if analysis['exists'] and not analysis['is_all_nan']:
                    simplified['key_columns_missing_ratio'][col] = analysis['nan_ratio']
        
        simplified_results.append(simplified)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_file}")


def assess_data_sources_batch(directory_path, batch_size=5):
    """Batch assess confidence level of data sources"""
    
    # Initialize LLM judge
    resultJudge = ResultJudge("deepseek-api")
    
    # Collect quality information for all files
    all_quality_infos = []
    valid_files = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            print(f"Analyzing: {filename}")
            
            quality_info = analyze_data_quality(file_path)
            if quality_info is not None:
                all_quality_infos.append(quality_info)
                valid_files.append(filename)
    
    if not all_quality_infos:
        print("No valid data files found")
        return []
    
    results = []
    
    # Process in batches
    for i in range(0, len(all_quality_infos), batch_size):
        batch = all_quality_infos[i:i+batch_size]
        batch_files = valid_files[i:i+batch_size]
        
        # Extract source names for this batch
        batch_source_names = [info['source_name'] for info in batch]
        print(f"\nEvaluating batch {i//batch_size + 1}: {', '.join(batch_source_names)}")
        
        # Generate batch prompt
        prompt = generate_batch_llm_prompt(batch, batch_size)

        print(prompt)
        
        try:
            # Call LLM for evaluation
            llm_response = resultJudge.judge(prompt)
            
            print(f"LLM response: {llm_response}")

            print(llm_response)
            
            # Parse LLM response (match using source names)
            confidence_dict = parse_llm_response(llm_response, batch)
            
            # Associate results with quality information
            for quality_info in batch:
                source_name = quality_info['source_name']
                confidence_level = confidence_dict.get(source_name, 3)  # Default level 3
                
                result = {
                    'file_name': quality_info['file_name'],
                    'source_name': source_name,
                    'quality_info': quality_info,
                    'confidence_level': confidence_level
                }
                results.append(result)
                print(f"Evaluation completed: {source_name} -> Level {confidence_level}")
            
            # Add delay to avoid API limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Batch evaluation failed: {str(e)}")
            # If batch evaluation fails, try individual evaluation
            for quality_info in batch:
                try:
                    single_prompt = generate_single_llm_prompt(quality_info)
                    llm_response = resultJudge.judge(single_prompt)
                    confidence_level = extract_single_confidence(llm_response)
                    
                    result = {
                        'file_name': quality_info['file_name'],
                        'source_name': quality_info['source_name'],
                        'quality_info': quality_info,
                        'confidence_level': confidence_level
                    }
                    results.append(result)
                    print(f"Individual evaluation completed: {quality_info['source_name']} -> Level {confidence_level}")
                    
                    time.sleep(1)  # Add delay for individual evaluation too
                    
                except Exception as e2:
                    print(f"Individual evaluation also failed for {quality_info['source_name']}: {str(e2)}")
                    # Assign default level
                    result = {
                        'file_name': quality_info['file_name'],
                        'source_name': quality_info['source_name'],
                        'quality_info': quality_info,
                        'confidence_level': 3  # Default medium
                    }
                    results.append(result)
    
    return results

def parse_llm_response(llm_response, batch_quality_infos):
    """Parse LLM response, extract JSON format confidence dictionary, match using source names"""
    try:
        # Try to parse JSON directly
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            response_dict = json.loads(json_str)
        else:
            # If no JSON found, try to extract key-value pairs from text
            response_dict = {}
            lines = llm_response.split('\n')
            for line in lines:
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        key = parts[0].strip().replace("'", "").replace('"', "")
                        value_str = parts[1].strip()
                        # Extract number
                        level_match = re.search(r'\d', value_str)
                        if level_match:
                            response_dict[key] = int(level_match.group())
        
        # Create confidence dictionary using source names as keys
        confidence_dict = {}
        batch_source_names = [info['source_name'] for info in batch_quality_infos]
        
        for source_name in batch_source_names:
            # Try multiple matching methods
            if source_name in response_dict:
                confidence_dict[source_name] = response_dict[source_name]
            else:
                # Try partial matching
                matched = False
                for key in response_dict.keys():
                    if key in source_name or source_name in key:
                        confidence_dict[source_name] = response_dict[key]
                        matched = True
                        break
                
                if not matched:
                    # If no match found, assign default level
                    print(f"Warning: Could not find matching key for source {source_name}. Keys in response: {list(response_dict.keys())}")
                    confidence_dict[source_name] = 3  # Default medium
        
        return confidence_dict
    except Exception as e:
        print(f"Failed to parse LLM response: {str(e)}")
        # Return a default dictionary with all sources at level 3
        batch_source_names = [info['source_name'] for info in batch_quality_infos]
        return {source_name: 3 for source_name in batch_source_names}

def generate_single_llm_prompt(quality_info):
    """Generate simplified prompt for single data source (backup)"""
    prompt = f"""
Movie data source: {quality_info['source_name']}
Total records: {quality_info['total_records']}
Please provide confidence level (1-5): 
"""
    return prompt

def extract_single_confidence(llm_response):
    """Extract level from single evaluation response"""
    level_match = re.search(r'[1-5]', llm_response)
    return int(level_match.group()) if level_match else 3

def main():
    """Main function"""
    directory_path = "/home/lwh/QueryFusion/data/dataset/movie_m/raw_data"
    
    if not os.path.exists(directory_path):
        print("Directory does not exist!")
        return
    
    batch_size = int(input("Enter number of files to evaluate per batch (recommended 5-10): ").strip() or "5")
    
    print("Starting batch analysis of movie data source quality...")
    results = assess_data_sources_batch(directory_path, batch_size)
    
    if results:
        save_results_to_file(results)
        
        # Print summary information
        print("\n=== Movie Data Source Confidence Summary ===")
        for result in results:
            print(f"{result['source_name']} ({result['file_name']}): Level {result['confidence_level']}")
            
        # Count number at each level
        from collections import Counter
        level_counts = Counter([r['confidence_level'] for r in results])
        print(f"\nLevel statistics: {dict(level_counts)}")
    else:
        print("No valid data files found or analysis failed.")

if __name__ == "__main__":
    main()