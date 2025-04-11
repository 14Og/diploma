import argparse


PARSER = argparse.ArgumentParser(description="Sun sensor calibration script.")
PARSER.add_argument(
    "-stage",
    "-s",
    type=str,
    dest="stage",
    help="Calibration stage.",
    required=False,
    choices=["first", "second", "FIRST", "SECOND"],
)
PARSER.add_argument(
    "-path",
    "-p",
    type=str,
    dest="path",
    help="Path to data in csv mode.",
    required=False,
)

PARSER.add_argument(
    "-mode",
    "-m",
    dest="mode",
    choices=["report", "plotter", "continue"],
    help="Mode to specify plotting options. Use 'release' to call from QT. \
    Use 'plotter' to perform some fancy plots after processing",
    default="continue",
    required=False,
)

PARSER.add_argument(
    "-panel",
    dest="panel",
    type=str,
    help="Panel number or name to distinguish one from another.",
    required=True,
)

PARSER.add_argument(
    "-i",
    "-interp",
    dest="interpolation",
    type=str,
    choices=["linear", "cubic"],
    default="linear",
    help="Interpolation method used in calibration process",
    required=False,
)


PARSER.add_argument("--reverse", action="store_true", dest="reverse")
PARSER.add_argument("--no-reverse", dest="reverse", action="store_false")
PARSER.set_defaults(reverse=False)


def check_args(args: argparse.Namespace):
    if vars(args)["mode"] == "report" and vars(args)["stage"] == "first":
        raise Exception(
            "Arguments for first stage calibration is 'plotter' or 'continue', not 'report'."
        )
    if vars(args)["mode"] != "plotter" and not vars(args)["stage"]:
        raise Exception(
            "You have to provide calibration stage with perfomance mode. Use 'plotter' for non-calibration actions."
        )
    if vars(args)["reverse"] and vars(args)["stage"] == "first":
        raise Exception("Reverse option is legal for second calibration stage.")


if __name__ == "__main__":
    raise Exception("Utility module.")
