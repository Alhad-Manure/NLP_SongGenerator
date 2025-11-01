import re
import os
from pathlib import Path

def extract_hindi_from_text(content):
    """
    Extract only Hindi (Devanagari) content from text.
    Removes English words, numbers, and special characters.
    """
    # Extract only Devanagari characters and essential punctuation
    # Devanagari Unicode range: \u0900-\u097F
    hindi_pattern = r'[^\u0900-\u097F\s।?]+'
    
    # Remove all non-Hindi content
    cleaned_content = re.sub(hindi_pattern, '', content)
    
    # Remove excessive whitespace and empty lines
    lines = cleaned_content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Strip whitespace and remove empty lines
        line = line.strip()
        if line:  # Only keep non-empty lines
            cleaned_lines.append(line)
    
    # Join lines with newlines
    return '\n'.join(cleaned_lines)

def process_directory(input_dir, output_file, file_extension='.txt'):
    """
    Process all text files in a directory and combine them into one file.
    Each file's content is separated by '--' and a newline.
    """
    try:
        # Get all files with specified extension
        input_path = Path(input_dir)
        files = sorted(input_path.glob(f'*{file_extension}'))
        
        if not files:
            print(f"No {file_extension} files found in '{input_dir}'")
            return
        
        print(f"Found {len(files)} files to process:\n")
        
        all_content = []
        processed_count = 0
        
        for file_path in files:
            print(f"Processing: {file_path.name}")
            
            try:
                # Read the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract Hindi content
                hindi_content = extract_hindi_from_text(content)
                
                if hindi_content:  # Only add if there's Hindi content
                    all_content.append(hindi_content)
                    processed_count += 1
                    print(f"  ✓ Extracted Hindi content")
                else:
                    print(f"  ⚠ No Hindi content found")
                    
            except Exception as e:
                print(f"  ✗ Error processing {file_path.name}: {e}")
        
        # Combine all content with separator
        combined_content = '\n--\n'.join(all_content)
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        print(f"\n{'='*50}")
        print(f"Successfully processed {processed_count} files!")
        print(f"Combined output saved to: {output_file}")
        print(f"{'='*50}")
        
        # Preview
        print("\n--- Preview of combined content (first 400 characters) ---")
        print(combined_content[:400] + "..." if len(combined_content) > 400 else combined_content)
        
    except Exception as e:
        print(f"Error: {e}")

# Usage
if __name__ == "__main__":
    # Specify your input directory and output file
    input_directory = "./IndianHindiSongsLyricsDataset"  # Current directory (change to your directory path)
    output_filename = "cleaned/all_hindi_combined2.txt"
    file_extension = ".txt"  # Change if you want to process other file types
    
    process_directory(input_directory, output_filename, file_extension)
