print("importing")
import pyvjoy, math, numpy as np, time
print("importing")
print('importing')
j = pyvjoy.VJoyDevice(1)
print(j)
JOY_AXIS_MIN = 1
JOY_AXIS_MAX = 32768//2
JOY_AXIS_CTR = 32768//2
j.set_axis(pyvjoy.HID_USAGE_Y, JOY_AXIS_CTR)
j.set_axis(pyvjoy.HID_USAGE_X, JOY_AXIS_CTR)
j.set_axis(pyvjoy.HID_USAGE_RY, JOY_AXIS_CTR)
j.set_axis(pyvjoy.HID_USAGE_RX, JOY_AXIS_CTR)
j.set_axis(pyvjoy.HID_USAGE_Z, JOY_AXIS_MIN)
j.set_axis(pyvjoy.HID_USAGE_RZ, JOY_AXIS_MIN)
for i in range(5):
    print(5-i)
    time.sleep(1)
while True:
    for i in range(JOY_AXIS_MIN, JOY_AXIS_CTR + JOY_AXIS_MAX):
        j.set_axis(pyvjoy.HID_USAGE_X, i)
    for i in range(JOY_AXIS_CTR + JOY_AXIS_MAX, JOY_AXIS_MIN-1, -1):
        j.set_axis(pyvjoy.HID_USAGE_X, i)
##    j.set_axis(pyvjoy.HID_USAGE_X, JOY_AXIS_CTR)
##    for i in range(JOY_AXIS_MIN, JOY_AXIS_CTR + JOY_AXIS_MAX):
##        j.set_axis(pyvjoy.HID_USAGE_Z, i)
##    for i in range(JOY_AXIS_CTR + JOY_AXIS_MAX, JOY_AXIS_MIN-1, -1):
##        j.set_axis(pyvjoy.HID_USAGE_Z, i)
##    for i in range(JOY_AXIS_MIN, JOY_AXIS_CTR + JOY_AXIS_MAX):
##        j.set_axis(pyvjoy.HID_USAGE_RZ, i)
##    for i in range(JOY_AXIS_CTR+ JOY_AXIS_MAX, JOY_AXIS_MIN-1, -1):
##        j.set_axis(pyvjoy.HID_USAGE_RZ, i)
##    for i in range(JOY_AXIS_MIN, JOY_AXIS_CTR + JOY_AXIS_MAX):
##        j.set_axis(pyvjoy.HID_USAGE_RY, i)
##    for i in range(JOY_AXIS_CTR + JOY_AXIS_MAX, JOY_AXIS_MIN-1, -1):
##        j.set_axis(pyvjoy.HID_USAGE_RY, i)
