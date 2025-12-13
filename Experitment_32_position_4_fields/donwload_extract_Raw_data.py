import gdown
import zipfile
import os

def download_with_gdown():
    """
    This code donwload the code raw data 
    """
    # directory actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    #  The link of the data
    url = "https://drive.google.com/file/d/1EiH7z0H6B-0s9SMiV39Z-7LTf51rcBUE/view?usp=sharing"
    
    # Extract file ID
    file_id = "1EiH7z0H6B-0s9SMiV39Z-7LTf51rcBUE"
    
  
    output = os.path.join(script_dir, "downloaded_file.zip")
    
    print(f"Script directory: {script_dir}")
    print(f"Downloading file ID: {file_id}")
    print(f"Saving as: {output}")
    
  
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
    except Exception as e:
        print(f"Download failed with error: {e}")
        print("Trying alternative method...")
        # Try alternative
        import requests
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url, stream=True)
        with open(output, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download completed with alternative method")
    
    # Check if file was downloaded
    if os.path.exists(output):
        file_size = os.path.getsize(output)
        print(f" Download successful! File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Extract
        print(f"\nExtracting {output} to {script_dir}...")
        try:
            with zipfile.ZipFile(output, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"Found {len(file_list)} files in archive")
                zip_ref.extractall(script_dir)
                print("✓ Extraction complete!")
                
                # List extracted files
                print("\nExtracted files in script directory:")
                for file in os.listdir(script_dir):
                    file_path = os.path.join(script_dir, file)
                    if file != "downloaded_file.zip" and file != os.path.basename(__file__):
                        if os.path.isfile(file_path):
                            size_kb = os.path.getsize(file_path) / 1024
                            print(f"  - {file} ({size_kb:.1f} KB)")
                        else:
                            print(f"  - {file}/ (folder)")
                
        except zipfile.BadZipFile:
            print(" Error: Downloaded file is not a valid zip file")
            print("The file might be corrupted or not a zip file")
        except Exception as e:
            print(f" Extraction error: {e}")
        
        # Remove zip file
        try:
            os.remove(output)
            print(f"\n✓ Removed {os.path.basename(output)}")
        except:
            print(f"\n⚠ Could not remove zip file")
    else:
        print(" Download failed - file not found!")
        print(f"Expected file at: {output}")
        
        # Show what's in the directory
        print(f"\nFiles in {script_dir}:")
        for f in os.listdir(script_dir):
            print(f"  - {f}")

if __name__ == "__main__":
    download_with_gdown()