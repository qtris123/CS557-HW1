"""Package the user's completed report, code, and exactly two baseline DQN logs."""
import argparse
import json
from pathlib import Path
import re
import zipfile
from run_experiments import is_complete, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--firstname", required=True)
    parser.add_argument("--lastname", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_root", default=".", type=Path)
    args = parser.parse_args()
    for value in (args.firstname, args.lastname):
        if not re.fullmatch(r"[A-Za-z][A-Za-z_-]*", value):
            parser.error("Use letters, hyphens, or underscores for the submission name")
    if not args.report.is_file() or args.report.suffix.lower() != ".pdf":
        parser.error("--report must point to your completed PDF report")
    if not args.report.read_bytes().startswith(b"%PDF-"):
        parser.error("The report is not a PDF file")
    if not all(value.lower() in args.report.stem.lower() for value in (args.firstname, args.lastname)):
        parser.error("Include your first and last names in the PDF filename")
    selected = []
    for env in ("LunarLander-v2", "CarRacing-v2"):
        directory = run_dir(args.output_root, env, "p1_dqn", args.seed)
        if not is_complete(directory):
            parser.error(f"Need a completed baseline DQN run: {directory}")
        config = json.loads((directory / "config.json").read_text())
        if config["use_double_dqn"] or config["dueling_dqn"] or config["use_per"]:
            parser.error(f"Expected standard DQN + uniform replay: {directory}")
        selected.append((env, directory))
    output = args.output_root / f"{args.firstname.lower()}_{args.lastname.lower()}_hw1.zip"
    source = Path(__file__).resolve().parent
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.report, args.report.name)
        for file in sorted(source.iterdir()):
            if file.is_file() and (file.suffix == ".py" or file.name.startswith("requirements") or file.name.endswith("GUIDE.md")):
                archive.write(file, f"code/{file.name}")
        for env, directory in selected:
            for file in sorted(directory.rglob("*")):
                if file.is_file():
                    archive.write(file, f"selected_runs/{env}/{file.relative_to(directory)}")
    print(f"Submission ZIP: {output.resolve()}")


if __name__ == "__main__":
    main()
