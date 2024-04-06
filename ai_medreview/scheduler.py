import time
import subprocess
import sys
from colorama import Fore, Style, init

init()  # Initializes Colorama

wait_time = 7200  # 2 hours

while True:
    subprocess.run(["python", "friendsfamilytest/data.py"])

    # Countdown for wait_time
    for remaining in range(wait_time, 0, -1):
        sys.stdout.write("\r")
        # Setting the color to bold red for the countdown
        countdown_message = "{0}{1}{2} seconds remaining. Friends & Family Test".format(
            Fore.RED + Style.BRIGHT, remaining, Style.RESET_ALL
        )
        sys.stdout.write(countdown_message)
        sys.stdout.flush()
        time.sleep(1)

    sys.stdout.write("\rExecuting script...                   \n")
    sys.stdout.flush()
