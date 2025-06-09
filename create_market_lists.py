import os
import glob

def create_list_file(image_dir, output_file):
    """Create a list file containing image names from the given directory."""
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    image_names = [os.path.basename(f) for f in image_files]
    # Sort to ensure consistent order
    image_names.sort()
    
    with open(output_file, 'w') as f:
        for name in image_names:
            # Only include if not junk images (-1)
            if not name.startswith('-1'):
                f.write(name + '\n')

def main():
    # Update these paths to match your Market-1501 dataset location
    market_root = "C:/Users/Haris_DFKI/Desktop/SnP/data/reid_data/market"
    
    # Create list files for train, query and gallery
    create_list_file(
        os.path.join(market_root, 'bounding_box_train'),
        os.path.join(market_root, 'name_train.txt')
    )
    create_list_file(
        os.path.join(market_root, 'query'),
        os.path.join(market_root, 'name_query.txt')
    )
    create_list_file(
        os.path.join(market_root, 'bounding_box_test'),
        os.path.join(market_root, 'name_test.txt')
    )
    print("Created list files successfully!")

if __name__ == '__main__':
    main() 