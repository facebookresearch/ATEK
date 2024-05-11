from setuptools import find_packages, setup


def main():
    setup(
        name="atek",
        version="0.2",
        description="Aria train and evaluation kits",
        author="Meta Reality Labs Research",
        packages=find_packages(),  # automatically discover all packages and subpackages
    )


if __name__ == "__main__":
    main()
