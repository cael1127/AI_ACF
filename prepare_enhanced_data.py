"""
Prepare enhanced dataset by combining multiple sources.
"""
import os
import glob

def combine_text_files(output_file='data/enhanced/combined.txt'):
    """Combine all text files into one training file."""
    patterns = [
        # Enhanced instruction/style files
        'data/enhanced/math_conversation.txt',
        'data/enhanced/coding_tutorial.txt',
        'data/enhanced/instruction_conversation.txt',
        'data/enhanced/instruction_math.txt',
        'data/enhanced/instruction_code.txt',
        # Any other enhanced txt files
        'data/enhanced/*.txt',
        # Cleaned scraped corpus
        'data/cleaned/*.txt',
        # Synthetic/local small corpora
        'data/training_data.txt',
        'data/synthetic/simple.txt',
    ]
    
    all_texts = []
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if len(content) > 100:  # Only add substantial content
                        all_texts.append(content)
                        print(f"Added: {file} ({len(content)} chars)")
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    # Write combined content
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n\n'.join(all_texts))
    
    total_chars = sum(len(text) for text in all_texts)
    print(f"\nCombined {len(all_texts)} files into {output_file}")
    print(f"Total characters: {total_chars}")
    return output_file

if __name__ == '__main__':
    combine_text_files()

