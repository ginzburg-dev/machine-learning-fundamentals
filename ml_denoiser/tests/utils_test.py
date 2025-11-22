import re

if __name__ == "__main__":
    STR = "name.0001.exr"
    match = re.search(r'\.(\d+)', STR)
    print(match.group(1))  # Output: 0001