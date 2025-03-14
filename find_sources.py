import re
import os
import sys
import argparse

def extract_function_names(log):
    """
    Extract function names from linker error log lines.
    It looks for lines with "undefined reference to `...'" and then:
      - Removes any parameters (anything from '(' onward).
      - Extracts the part after the final '::'.
    """
    signatures = re.findall(r"undefined reference to `(.*?)'", log)
    names = []
    for sig in signatures:
        # Remove function parameters if present.
        name_only = sig.split('(')[0]
        # Take the part after the final "::"
        final_name = name_only.split("::")[-1]
        names.append(final_name)
    # Remove duplicates while preserving order.
    unique_names = list(dict.fromkeys(names))
    return unique_names

def find_definition(function_name, search_dir, file_extensions):
    """
    Search for the function definition in files under search_dir.
    The heuristic uses a regex that looks for the function name followed
    by a parameter list and an opening curly brace '{'.
    """
    matching_files = []
    # Regex to match a function definition pattern.
    pattern = r"\b" + re.escape(function_name) + r"\s*\([^;]*\)\s*\{"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    for root, _, files in os.walk(search_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in file_extensions:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if regex.search(content):
                            matching_files.append(file_path)
                except Exception:
                    continue
    return matching_files

def main():
    parser = argparse.ArgumentParser(
        description="Extract function names from log and search for their definitions."
    )
    parser.add_argument("log_file", nargs="?", help="Path to the log file. If not provided, reads from stdin.")
    parser.add_argument("--search-dir", default=".", help="Directory to search for function definitions (default: current directory)")
    args = parser.parse_args()

    # Read log text either from piped input or a file argument.
    if not sys.stdin.isatty():
        log_text = sys.stdin.read()
    elif args.log_file:
        try:
            with open(args.log_file, "r") as file:
                log_text = file.read()
        except Exception as e:
            print(f"Error reading file {args.log_file}: {e}")
            sys.exit(1)
    else:
        print("No log input provided. Provide a log file or pipe input.")
        sys.exit(1)

    # Extract function names (duplicates removed).
    function_names = extract_function_names(log_text)
    print("Extracted unique function names (only the part after the last '::'):")
    for name in function_names:
        print(f"  {name}")
    
    # Define allowed source file extensions.
    file_extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".S", ".s"}

    print("\nSearching for definitions in directory:", args.search_dir)
    for func in function_names:
        matching_files = find_definition(func, args.search_dir, file_extensions)
        if matching_files:
            print(f"\n{func}:")
            for f in matching_files:
                print(f"  {f}")
        else:
            print(f"\n{func}: Definition not found")

if __name__ == "__main__":
    main()

