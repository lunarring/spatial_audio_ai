import numpy as np
import pytest
from tools.playback_stream import SoundNetworkStreamer, BLOCKSIZE, BlackHoleStereoRelayer

class DummySimulatedSocket:
    """
    A dummy simulated socket to mimic SimulatedSocket behavior.
    """
    def __init__(self):
        self.connected = False
        self.last_data = None

    def connect(self, addr):
        self.connected = True

    def sendall(self, data):
        if not self.connected:
            raise Exception("Not connected")
        self.last_data = data

    def recv(self):
        if not self.connected:
            raise Exception("Not connected")
        return self.last_data

    def close(self):
        self.connected = False

# Monkeypatch fixture to override the socket in SoundNetworkStreamer with our dummy
@pytest.fixture
def simulated_streamer(monkeypatch):
    # Create an instance with simulation enabled
    streamer = SoundNetworkStreamer(simulate=True)
    # Override the socket with our dummy simulated socket
    dummy_socket = DummySimulatedSocket()
    dummy_socket.connect((streamer.host, streamer.port))
    streamer.socket = dummy_socket
    streamer.socket_connected = True
    return streamer

def test_send_and_receive(simulated_streamer):
    # Create a dummy numpy array with shape (2, BLOCKSIZE*2) since BLOCKSIZE divides the second dim
    dummy_data = np.random.randn(2, BLOCKSIZE * 2).astype(np.float32)
    # Use send_and_receive and expect echo behavior
    echoed = simulated_streamer.send_and_receive(dummy_data)
    # Verify that the echoed data is the same as sent data
    np.testing.assert_array_equal(echoed, dummy_data)

def test_toggle_playback_mode(monkeypatch):
    # Override get_device_index_by_name to bypass audio device query during testing
    monkeypatch.setattr(BlackHoleStereoRelayer, 'get_device_index_by_name', lambda self, device_name: 0)
    # Create the relayer with initial mode 'alternating'
    relayer = BlackHoleStereoRelayer(mapping_scheme='alternating')
    assert relayer.mapping_scheme == 'alternating'
    # Simulate pressing 'T' to toggle mode
    relayer.handle_key_press('T')
    assert relayer.mapping_scheme == 'grouped'
    # Toggle back to alternating
    relayer.handle_key_press('T')
    assert relayer.mapping_scheme == 'alternating'

if __name__ == "__main__":
    pytest.main([__file__])