import pyautogui
from screeninfo import get_monitors

pyautogui.FAILSAFE = False

# Get all monitor information
monitors = get_monitors()

# Assume the external monitor is the second monitor (index 1)
external_monitor = monitors[1]

# Get the resolution of the external monitor
external_monitor_width = external_monitor.width
external_monitor_height = external_monitor.height

# Calculate the desired position on the external monitor (e.g., center)
x_position_on_external_monitor = external_monitor_width - 200
y_position_on_external_monitor = external_monitor_height - 1080

# Ensure the calculated position is within the bounds of the external monitor
x_position_on_external_monitor = min(max(x_position_on_external_monitor, 0), external_monitor_width)
y_position_on_external_monitor = min(max(y_position_on_external_monitor, 0), external_monitor_height)

# Move the mouse to the calculated position on the external monitor
pyautogui.moveTo(external_monitor.x + x_position_on_external_monitor, 
                  external_monitor.y + y_position_on_external_monitor, duration=0.2)
