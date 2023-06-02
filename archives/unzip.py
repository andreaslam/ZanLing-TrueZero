import zipfile

def unzip_file(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print("File extracted successfully!")

# Example usage
zip_file_path = './chess_games.db.zip'
extract_directory = './chess_games.db'
unzip_file(zip_file_path, extract_directory)
