from pathlib import Path


def cleanup(path: Path):
	if not path.exists():
		return

	if not path.is_dir():
		path.unlink()
		return

	for file in path.iterdir():
		cleanup(file)
	path.rmdir()
