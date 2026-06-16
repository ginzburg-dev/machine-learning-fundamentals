import re

if __name__ == "__main__":
    STR = "name.0001.exr"
    match = re.search(r'\.(\d+)', STR)
    if match is None:
        print("No numeric sequence found in string")
    else:
        print(match.group(1))
