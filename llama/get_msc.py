import subprocess

subprocess.run(["pip", "install", "parlai"])
subprocess.run(["parlai", "display_data", "-t", "msc"])
