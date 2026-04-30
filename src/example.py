import rbpodo as rb
import numpy as np


def _main():
    robot = rb.Cobot("192.168.1.10")
    rc = rb.ResponseCollector()

    try:
        robot.set_operation_mode(rc, rb.OperationMode.Real)
        rc = rc.error().throw_if_not_empty()

        target_point = np.array([-120, -447, 369, 90, 0, 44])
        robot.move_l(rc, target_point, 300, 400, rb.ReferenceFrame.Base)
        rc = rc.error().throw_if_not_empty()

        if robot.wait_for_move_started(rc, 0.5).is_success():
            robot.wait_for_move_finished(rc)
        rc = rc.error().throw_if_not_empty()
    finally:
        print('Exit')


if __name__ == '__main__':
    _main()