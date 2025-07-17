#!/usr/bin/env python3
"""
Minimal ZMQ connection test to isolate the issue
"""

import time

def test_zmq_basic():
    """Test basic ZMQ connection without audio"""
    print("Testing basic ZMQ connection...")
    
    try:
        import lunar_tools as lt
        print("✅ lunar_tools imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import lunar_tools: {e}")
        return
    
    # Test server creation
    try:
        print("Creating ZMQ server...")
        server = lt.ZMQPairEndpoint(is_server=True, ip='10.40.49.47', port='5556')
        print("✅ ZMQ server created")
    except Exception as e:
        print(f"❌ Failed to create ZMQ server: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test client creation
    try:
        print("Creating ZMQ client...")
        client = lt.ZMQPairEndpoint(is_server=False, ip='10.40.49.47', port='5556')
        print("✅ ZMQ client created")
    except Exception as e:
        print(f"❌ Failed to create ZMQ client: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test basic communication
    try:
        print("Testing basic communication...")
        
        # Client sends message
        test_msg = {"test": "hello", "client_id": "test123"}
        client.send_json(test_msg)
        print("✅ Client sent message")
        
        # Give it a moment
        time.sleep(0.1)
        
        # Server checks for messages
        messages = server.get_messages()
        print(f"Server received {len(messages)} messages")
        
        if messages:
            print(f"✅ Message received: {messages[0]}")
        else:
            print("❌ No messages received")
            
        # Test server sending back
        server.send_json({"response": "hello back"})
        time.sleep(0.1)
        
        client_messages = client.get_messages()
        print(f"Client received {len(client_messages)} messages")
        if client_messages:
            print(f"✅ Response received: {client_messages[0]}")
        else:
            print("❌ No response received")
            
    except Exception as e:
        print(f"❌ Communication test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zmq_basic() 