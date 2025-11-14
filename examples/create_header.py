import os

LICENSE_FILE = "/home/abosman/dev/VERONA/license_header.txt"
PROJECT_DIR = "."  # Change this if your source is in a subdirectory, e.g. "src"

# Read the license header text
with open(LICENSE_FILE, "r") as f:
    license_text = f.read().strip() + "\n\n"

for root, _, files in os.walk(PROJECT_DIR):
    for filename in files:
        if filename.endswith(".py") and filename != os.path.basename(__file__):
            path = os.path.join(root, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Skip if header already exists
            if license_text.strip() in content:
                continue

            # Avoid putting header before shebang
            if content.startswith("#!"):
                lines = content.splitlines(True)
                new_content = lines[0] + license_text + "".join(lines[1:])
            else:
                new_content = license_text + content

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"Added license to {path}")
