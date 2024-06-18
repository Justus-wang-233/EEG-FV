import importlib.metadata


def generate_requirements():
    requirements = []
    for dist in importlib.metadata.distributions():
        # Get the package name and version
        package = f"{dist.metadata['Name']}~={dist.version}"
        requirements.append(package)

    # Write to requirements.txt
    with open('requirements.txt', 'w') as f:
        for req in sorted(requirements):
            f.write(req + '\n')


if __name__ == "__main__":
    generate_requirements()

