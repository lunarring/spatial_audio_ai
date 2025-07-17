#!/usr/bin/env python3
"""
Simple test script to verify ZMQ connection behavior when server is down.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from spatial_audio_ai.tools.client import SoundNetworkStreamer

def test_zmq_connection_when_server_down():
    print("Testing ZMQ connection when server is down...")
    print("=" * 50)
    
    try:
        # This should fail because no server is running
        streamer = SoundNetworkStreamer(
            host="10.40.49.47", 
            port=9999, 
            zmq_port=5556,
            profile='stable'  # This triggers ZMQ usage
        )
        print("❌ ERROR: Connection should have failed!")
        return False
        
    except Exception as e:
        print(f"✅ SUCCESS: Connection properly failed with: {e}")
        return True

if __name__ == "__main__":
    success = test_zmq_connection_when_server_down()
    if success:
        print("\n✅ Test passed: ZMQ connection properly detects when server is down")
    else:
        print("\n❌ Test failed: ZMQ connection did not detect server absence")
    
    sys.exit(0 if success else 1) 