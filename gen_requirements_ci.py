import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Install TorchUtils dependencies.')
    # first read latest requirements file.
    req = open('requirements.txt').read().splitlines()
    # package:version
    req = {line.split("==")[0]: line.split("==")[1] for line in req}
    for pkg, pkg_version in req.items():
        parser.add_argument('--' + pkg, type=str, default=pkg_version,
                            help=pkg + ' version')
    parser.add_argument('--skip', type=str, nargs='+')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # find packages to be skipped
    skip_pkgs = {}
    if args.skip is not None:
        skip_pkgs = set(args.skip)
    # generate requirements_ci.txt
    with open('requirements_ci.txt', 'w') as f:
        total = len(args.__dict__) - len(skip_pkgs) - 1
        count = 1
        for pkg, pkg_version in args.__dict__.items():
            if pkg not in skip_pkgs and pkg != "skip":
                if (count == total):
                    f.write(pkg + '==' + pkg_version)
                else:
                    f.write(pkg + '==' + pkg_version + '\n')
                count += 1
