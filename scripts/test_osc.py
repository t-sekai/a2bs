from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher

# Define the message handler function
def message_handler(address, *args):
    print(f"Received OSC message at address {address}: {args}")

# Create an OSC server listening on 127.0.0.1:5008
dispatcher = Dispatcher()
dispatcher.map("/tongueOut", print)
server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 5008), dispatcher)

# Start the server
print("OSC Server listening on 127.0.0.1:5008...")
server.serve_forever()
