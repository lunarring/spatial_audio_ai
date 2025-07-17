#!/usr/bin/env python3
"""
Network debugging script to identify connection issues
"""

import socket
import subprocess
import platform
import time

def check_network_connectivity():
    """Check various network connectivity aspects"""
    print("=== NETWORK DEBUGGING ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # 1. Check if server host is reachable
    server_host = "10.40.49.47"
    udp_port = 9999
    zmq_port = 5556
    
    print(f"\n1. Testing reachability to {server_host}...")
    
    # Ping test
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(['ping', '-c', '3', server_host], 
                                  capture_output=True, text=True, timeout=10)
        else:  # Linux
            result = subprocess.run(['ping', '-c', '3', server_host], 
                                  capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ Ping to {server_host} successful")
        else:
            print(f"❌ Ping to {server_host} failed")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"❌ Ping test failed: {e}")
    
    # 2. Check UDP port connectivity
    print(f"\n2. Testing UDP port {udp_port}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        sock.connect((server_host, udp_port))
        print(f"✅ UDP connection to {server_host}:{udp_port} successful")
        sock.close()
    except Exception as e:
        print(f"❌ UDP connection failed: {e}")
    
    # 3. Check ZMQ port with raw TCP
    print(f"\n3. Testing ZMQ port {zmq_port} (raw TCP)...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((server_host, zmq_port))
        if result == 0:
            print(f"✅ TCP connection to {server_host}:{zmq_port} successful")
        else:
            print(f"❌ TCP connection to {server_host}:{zmq_port} failed (error code: {result})")
        sock.close()
    except Exception as e:
        print(f"❌ TCP connection test failed: {e}")
    
    # 4. Check local network interfaces
    print(f"\n4. Local network interfaces...")
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"Hostname: {hostname}")
        print(f"Local IP: {local_ip}")
        
        # Get all network interfaces
        import subprocess
        if platform.system() == "Darwin":
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
        else:
            result = subprocess.run(['ip', 'addr'], capture_output=True, text=True)
        
        print("Network interfaces:")
        print(result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout)
        
    except Exception as e:
        print(f"❌ Network interface check failed: {e}")
    
    # 5. Test alternative connection methods
    print(f"\n5. Testing alternative hosts...")
    alternative_hosts = ['localhost', '127.0.0.1', local_ip if 'local_ip' in locals() else '127.0.0.1']
    
    for host in alternative_hosts:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, zmq_port))
            if result == 0:
                print(f"✅ Connection to {host}:{zmq_port} successful")
            else:
                print(f"❌ Connection to {host}:{zmq_port} failed")
            sock.close()
        except Exception as e:
            print(f"❌ Connection to {host}:{zmq_port} error: {e}")

def test_zmq_localhost():
    """Test ZMQ connection using localhost"""
    print(f"\n=== TESTING ZMQ ON LOCALHOST ===")
    
    try:
        import lunar_tools as lt
        print("✅ lunar_tools imported")
        
        # Test with localhost
        print("Testing ZMQ server on localhost...")
        server = lt.ZMQPairEndpoint(is_server=True, ip='127.0.0.1', port='5557')  # Different port
        print("✅ ZMQ server created on localhost:5557")
        
        time.sleep(0.5)  # Give server time to bind
        
        print("Testing ZMQ client on localhost...")
        client = lt.ZMQPairEndpoint(is_server=False, ip='127.0.0.1', port='5557')
        print("✅ ZMQ client created")
        
        # Test communication
        test_msg = {"test": "localhost_test"}
        client.send_json(test_msg)
        time.sleep(0.1)
        
        messages = server.get_messages()
        if messages:
            print(f"✅ Localhost ZMQ communication works: {messages[0]}")
        else:
            print("❌ No messages received on localhost")
            
    except Exception as e:
        print(f"❌ ZMQ localhost test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_network_connectivity()
    test_zmq_localhost() 