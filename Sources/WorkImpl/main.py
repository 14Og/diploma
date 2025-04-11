from Utils import Calibration
from Utils import PARSER, check_args
import sys

"""
arguments for first calibration:

(-p) - path to first csv log file. You're entitled not to provide it if all the files are present in calibration path.
(-panel) - panel name (number).
(-stage=first,FIRST) - calibration  stage.
(-interpolation=linear,cubic) - interpolation type for calibrator. Defaults to linear
-mode is defaults to continue.
Warning: fov_data.csv have to be present in calibration directory before first stage or FOV will be set to 70 degrees.


arguments for second calibration:

(-p) - path to second csv log file. You're entitled not to provide it if all the files are present in calibration path.
(-panel) - panel title (number).
(-stage=second,SECOND) - calibration stage
(-mode=report, continue) - depends on what you want. Defaults to continue
(-interpolation=linear) - interpolation type for calibrator. Linear or cubic. Defaults to linear.


arguments for plotter mode:
(-panel) - panel title (number).
(-mode=plotter) - to activate plotter mode.
"""

if __name__ == "__main__":
    args = PARSER.parse_args()
    check_args(args)
    c = Calibration(args.__dict__)
    c.emit()
