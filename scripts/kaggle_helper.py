import base64
import subprocess
from pathlib import Path
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
token = secrets.get_secret("GITHUB_TOKEN")

username = "AnikMallick"
repo_name = "bn-proverbs-in-context"

repo_dir = Path("/kaggle/working") / repo_name

auth = base64.b64encode(
    f"{username}:{token}".encode()
).decode()

try:
    subprocess.run(
        [
            "git",
            "-c",
            f"http.https://github.com/.extraheader="
            f"AUTHORIZATION: basic {auth}",
            "clone",
            f"https://github.com/{username}/{repo_name}.git",
            str(repo_dir),
        ],
        check=True,
    )
finally:
    del token
    del auth

print("Repository cloned successfully.")



# %cd /kaggle/working/bn-proverbs-in-context

# !pip install -e .

