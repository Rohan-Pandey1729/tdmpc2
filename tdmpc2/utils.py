import os

def prompt_if_file_exists(fp: str) -> bool:
    if os.path.exists(fp):
        while True:
            response = input(f"The file '{fp}' already exists. Do you want to overwrite it? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please respond with 'y' or 'n'.")
    else:
        return True
