# client_streaming_improved.py
import cv2
import socket
import struct
import pickle
import numpy as np

# Settings
host = '192.168.1.15'  # Replace with Pi's IP
port = 5000

# Create socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, port))
print(f"Connected to {host}:{port}")

# Payload size (4 bytes)
payload_size = struct.calcsize(">L")

data = b""

try:
    while True:
        # Retrieve message size
        while len(data) < payload_size:
            data += client_socket.recv(4096)
            
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        
        # Retrieve all frame data
        while len(data) < msg_size:
            data += client_socket.recv(4096)
            
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        try:
            # Deserialize and decode frame
            buffer = pickle.loads(frame_data)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
            if frame is not None:
                cv2.imshow("Live Stream", frame)
                
            if cv2.waitKey(1) == ord('q'):
                break
        except Exception as e:
            print(f"Frame processing error: {e}")
            
except KeyboardInterrupt:
    print("Closing connection")
except Exception as e:
    print(f"Connection error: {e}")
finally:
    client_socket.close()
    cv2.destroyAllWindows()
