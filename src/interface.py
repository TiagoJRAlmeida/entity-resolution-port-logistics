def select_files():
    import tkinter as tk
    from tkinter import filedialog
    
    # Set up the main application window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog and return a list of selected file paths
    file_paths = filedialog.askopenfilenames(title="Select standard file(s)")
    
    if file_paths:
        print("Selected file(s):")
        for path in file_paths:
            print(path)
        return file_paths        
    else:
        print("No files selected.")
        return set()
    

def is_valid_dataset(file_path):
    """
    Check if the file can be read into a DataFrame and contains required columns.
    Extra columns are allowed.
    Returns True if valid, False otherwise.
    """
    REQUIRED_COLUMNS = {"name", "identification_number"}
    import pandas as pd

    try:
        if file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, usecols=["name", "identification_number"], dtype={"name": str, "identification_number": str})
            except (pd.errors.ParserError, ValueError):
                df = pd.read_csv(file_path, delimiter=';', usecols=["name", "identification_number"], dtype={"name": str, "identification_number": str})
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path, usecols=["name", "identification_number"], dtype={"name": str, "identification_number": str})
        else:
            print(f"[ERROR] Unsupported file type: {file_path}")
            return False

        # Check that required columns are present, allow any extras
        if not REQUIRED_COLUMNS.issubset(df.columns):
            missing = REQUIRED_COLUMNS - set(df.columns)
            print(f"[ERROR] File '{file_path}' is missing required column(s): {', '.join(missing)}")
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Could not read file '{file_path}': {e}")
        return False


def display_menu(standard_datasets, datasets_to_standardize):
    """
    Displays the interactive menu and handles user input.
    Validates dataset formats and required columns.
    """
    from colorama import Fore, Style, init
    init(autoreset=True)

    def print_header():
        print(Fore.CYAN + Style.BRIGHT + "\n" + "#" * 60)
        print(Fore.CYAN + Style.BRIGHT + "#{:^58}#".format("DATASET STANDARDIZATION TOOL"))
        print(Fore.CYAN + Style.BRIGHT + "#" * 60)

    while True:
        print_header()
        print(Fore.YELLOW + Style.BRIGHT + "\nMain Menu:")
        print(Fore.GREEN + "\t1 - Insert Standard Data Sets")
        print(Fore.GREEN + "\t2 - Insert Data Sets to be Standardized")
        print(Fore.GREEN + "\t3 - Start Program")
        print(Fore.RED + "\t4 - Exit")

        try:
            choice = int(input(Fore.CYAN + Style.BRIGHT + "\nChoice: "))
            print()
            if choice == 1:
                print(Fore.MAGENTA + "--- Insert Standard Data Sets ---")
                new_files = select_files()
                valid_files = {path for path in new_files if is_valid_dataset(path)}
                if valid_files:
                    standard_datasets.update(valid_files)
                    print(Fore.GREEN + f"✔ Added {len(valid_files)} valid standard dataset(s). Total: {len(standard_datasets)}")
                else:
                    print(Fore.YELLOW + "⚠ No valid files selected.")
            elif choice == 2:
                print(Fore.MAGENTA + "--- Insert Data Sets to be Standardized ---")
                new_files = select_files()
                valid_files = {path for path in new_files if is_valid_dataset(path)}
                if valid_files:
                    datasets_to_standardize.update(valid_files)
                    print(Fore.GREEN + f"✔ Added {len(valid_files)} valid dataset(s) to be standardized. Total: {len(datasets_to_standardize)}")
                else:
                    print(Fore.YELLOW + "⚠ No valid files selected.")
            elif choice == 3:
                if not datasets_to_standardize:
                    print(Fore.RED + "[ERROR] You must add at least one dataset to be standardized before starting the program.")
                else:
                    print(Fore.GREEN + "\n▶ Starting the program...")
                    return standard_datasets, datasets_to_standardize
            elif choice == 4:
                print(Fore.BLUE + "Exiting the program. Goodbye!")
                exit(0)
            else:
                print(Fore.RED + "[ERROR] Invalid choice. Please select a valid option (1–4).")
        except ValueError:
            print(Fore.RED + "[ERROR] Invalid input. Please enter a number between 1 and 4.")