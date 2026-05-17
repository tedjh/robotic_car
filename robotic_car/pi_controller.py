# pi_controller.py
import msvcrt
import socket
import time

ARDUINO_IP = "192.168.0.38"  # Replace with Arduino's printed IP
ARDUINO_PORT = 80

COMMANDS = {
    "w": "W",  # forward
    "r": "R",  # backward
    "a": "A",  # left
    "d": "D",  # right
    "s": "S",  # stop
    "q": None,  # quit
}


def get_keypress():
    """Read a single keypress without needing Enter (Windows)."""
    return msvcrt.getwch().lower()


def connect(ip, port):
    """Keep trying to connect until successful."""
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(5)
            s.connect((ip, port))
            s.settimeout(None)
            print(f"Connected to Arduino at {ip}:{port}")
            return s
        except (socket.error, OSError) as e:
            print(f"Connection failed ({e}), retrying in 3s...")
            s.close()
            time.sleep(3)


def send_command(sock, cmd):
    """Send a command, return False if the connection is lost."""
    try:
        sock.sendall(cmd.encode())
        response = sock.recv(16).decode().strip()
        print(f"  → {response}")
        return True
    except (socket.error, OSError) as e:
        print(f"Send failed: {e}")
        return False


def main():
    print("Connecting...")
    print("Controls: W=forward  S=backward  A=left  D=right  SPACE=stop  Q=quit")

    sock = connect(ARDUINO_IP, ARDUINO_PORT)

    while True:
        key = get_keypress().lower()

        if key not in COMMANDS:
            print(f"Invalid key: {key}. Use W/A/S/D for movement, Q to quit.")
            continue

        if COMMANDS[key] is None:  # quit
            print("Quitting.")
            send_command(sock, "S")  # stop motors before exit
            sock.close()
            break

        cmd = COMMANDS[key]
        print(f"Key: {key!r}  →  Sending: {cmd!r}")

        ok = send_command(sock, cmd)
        if not ok:
            print("Lost connection, reconnecting...")
            sock.close()
            sock = connect(ARDUINO_IP, ARDUINO_PORT)


if __name__ == "__main__":
    main()
