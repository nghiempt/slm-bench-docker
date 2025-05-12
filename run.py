import json
import subprocess


def main():
    with open("main.json") as f:
        commands = json.load(f)
    for i, cmd in enumerate(commands, 1):
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
