import json
import sys
from collections import Counter

def analyze_json_structure(file_path):
    """
    Analyze and display the structure of a JSON dataset file.
    """
    print("=" * 80)
    print("JSON Dataset Analysis")
    print("=" * 80)
    
    try:
        print(f"\nReading file: {file_path}")
        print("This may take a moment for large files...\n")
        
        # Read and parse JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine the structure type
        print("=" * 80)
        print("DATA STRUCTURE ANALYSIS")
        print("=" * 80)
        
        if isinstance(data, dict):
            print("\n✓ Data Type: Dictionary/Object")
            print(f"  Number of top-level keys: {len(data)}")
            print(f"\n  Top-level keys:")
            for key in list(data.keys())[:10]:  # Show first 10 keys
                print(f"    - {key}")
            if len(data) > 10:
                print(f"    ... and {len(data) - 10} more keys")
            
            # Analyze first few entries
            if data:
                first_key = list(data.keys())[0]
                first_value = data[first_key]
                print(f"\n  Sample entry (key: '{first_key}'):")
                print(f"    Type: {type(first_value).__name__}")
                if isinstance(first_value, dict):
                    print(f"    Keys: {list(first_value.keys())}")
                elif isinstance(first_value, list):
                    print(f"    List length: {len(first_value)}")
                    if first_value:
                        print(f"    First item type: {type(first_value[0]).__name__}")
                        if isinstance(first_value[0], dict):
                            print(f"    First item keys: {list(first_value[0].keys())}")
        
        elif isinstance(data, list):
            print("\n✓ Data Type: Array/List")
            print(f"  Total number of records: {len(data):,}")
            
            if len(data) > 0:
                first_item = data[0]
                print(f"\n  First record type: {type(first_item).__name__}")
                
                if isinstance(first_item, dict):
                    print(f"  Number of fields per record: {len(first_item)}")
                    print(f"\n  Top-level field names:")
                    for key in first_item.keys():
                        value = first_item[key]
                        value_type = type(value).__name__
                        value_preview = str(value)[:50] if value else "None"
                        if len(str(value)) > 50:
                            value_preview += "..."
                        print(f"    - {key}: {value_type} (sample: {value_preview})")
                    
                    # If there's a _source field, analyze it in detail
                    if '_source' in first_item and isinstance(first_item['_source'], dict):
                        print(f"\n  Detailed structure of '_source' field (actual complaint data):")
                        source = first_item['_source']
                        print(f"    Number of fields in _source: {len(source)}")
                        print(f"    Fields in _source:")
                        for key in source.keys():
                            value = source[key]
                            value_type = type(value).__name__
                            if isinstance(value, str):
                                value_preview = value[:80] if value else "None"
                                if len(value) > 80:
                                    value_preview += "..."
                            else:
                                value_preview = str(value)[:50] if value else "None"
                                if len(str(value)) > 50:
                                    value_preview += "..."
                            print(f"      - {key}: {value_type} (sample: {value_preview})")
        else:
            print(f"\n✓ Data Type: {type(data).__name__}")
        
        # Display sample data
        print("\n" + "=" * 80)
        print("SAMPLE DATA")
        print("=" * 80)
        
        if isinstance(data, list):
            print(f"\nShowing first {min(3, len(data))} records:\n")
            for i, record in enumerate(data[:3], 1):
                print(f"Record {i}:")
                if isinstance(record, dict):
                    # Show metadata fields first
                    metadata_fields = ['_index', '_type', '_id', '_score']
                    for key in metadata_fields:
                        if key in record:
                            print(f"  {key}: {record[key]}")
                    
                    # Show _source field in detail
                    if '_source' in record and isinstance(record['_source'], dict):
                        print(f"\n  _source (actual complaint data):")
                        source = record['_source']
                        for key, value in source.items():
                            value_str = str(value)
                            if len(value_str) > 150:
                                value_str = value_str[:150] + "..."
                            print(f"    {key}: {value_str}")
                    
                    # Show any other fields
                    other_fields = {k: v for k, v in record.items() if k not in metadata_fields + ['_source']}
                    if other_fields:
                        for key, value in other_fields.items():
                            value_str = str(value)
                            if len(value_str) > 100:
                                value_str = value_str[:100] + "..."
                            print(f"  {key}: {value_str}")
                else:
                    print(f"  {record}")
                print()
        
        elif isinstance(data, dict):
            print(f"\nShowing first {min(3, len(data))} entries:\n")
            for i, (key, value) in enumerate(list(data.items())[:3], 1):
                print(f"Entry {i} (key: '{key}'):")
                if isinstance(value, dict):
                    for k, v in list(value.items())[:10]:
                        value_str = str(v)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"  {k}: {value_str}")
                    if len(value) > 10:
                        print(f"  ... and {len(value) - 10} more fields")
                elif isinstance(value, list):
                    print(f"  List with {len(value)} items")
                    if value:
                        print(f"  First item: {value[0]}")
                else:
                    print(f"  {value}")
                print()
        
        # Summary statistics
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Analyze field statistics
            all_keys = set()
            source_keys = set()
            
            for record in data[:1000]:  # Sample first 1000 records
                if isinstance(record, dict):
                    all_keys.update(record.keys())
                    if '_source' in record and isinstance(record['_source'], dict):
                        source_keys.update(record['_source'].keys())
            
            print(f"\nTotal records: {len(data):,}")
            print(f"Top-level fields: {', '.join(sorted(all_keys))}")
            
            if source_keys:
                print(f"\nFields in '_source' (actual data fields): {len(source_keys)}")
                print(f"  {', '.join(sorted(source_keys))}")
                
                # Check for classification-related fields
                classification_fields = ['label', 'category', 'class', 'target', 'complaint', 'text', 'description', 'summary', 'issue', 'product', 'sub_product', 'sub_issue']
                found_fields = [field for field in classification_fields if any(field.lower() in key.lower() for key in source_keys)]
                if found_fields:
                    print(f"\n✓ Potential classification/target fields found:")
                    for field in found_fields:
                        matching_keys = [key for key in source_keys if field.lower() in key.lower()]
                        print(f"    - {field}: {', '.join(matching_keys)}")
        
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        
    except json.JSONDecodeError as e:
        print(f"\n❌ Error: Invalid JSON format")
        print(f"   {str(e)}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    file_path = "complaints-2021-05-14_08_16.json"
    analyze_json_structure(file_path)

