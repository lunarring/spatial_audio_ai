import pygame
import numpy as np
import time
import threading
from spatial_audio_ai import (
    SO_Playback, 
    Spatializer, 
    Scene,
    SoundNetworkStreamer,
    get_sample_rate
)
from spatial_audio_ai.tools.spatializer import CHUNKSIZE, SAMPLING_RATE

# Initialize pygame
pygame.init()

# Window settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
SPEAKER_RADIUS = 8
SOUND_RADIUS = 12

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

# Global variables for mouse position sharing between threads
mouse_x, mouse_y = 0, 0
running = True

def _real_to_screen_coords(real_coords):
    """Convert real-world coordinates to screen coordinates"""
    # Real coordinates range approximately from -5 to 5 in both x and y
    # Map to screen coordinates with some padding
    padding = 50
    x_scale = (WINDOW_WIDTH - 2 * padding) / 10  # 10 unit range (-5 to 5)
    y_scale = (WINDOW_HEIGHT - 2 * padding) / 10
    
    screen_coords = []
    for coord in real_coords:
        x = padding + (coord[0] + 5) * x_scale
        y = padding + (5 - coord[1]) * y_scale  # Flip Y axis for screen coordinates
        screen_coords.append((int(x), int(y)))
    
    return screen_coords

def _screen_to_real_coords(screen_x, screen_y):
    """Convert screen coordinates to real-world coordinates"""
    padding = 50
    x_scale = (WINDOW_WIDTH - 2 * padding) / 10
    y_scale = (WINDOW_HEIGHT - 2 * padding) / 10
    
    real_x = (screen_x - padding) / x_scale - 5
    real_y = 5 - (screen_y - padding) / y_scale  # Flip Y axis back
    
    return np.array([real_x, real_y])

def pygame_thread():
    """Handle pygame events and drawing in a separate thread"""
    global mouse_x, mouse_y, running
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Interactive Sine Wave Positioning - Move mouse to control sound position")
    clock = pygame.time.Clock()
    
    # Get speaker positions for visualization
    spatializer = Spatializer()
    speaker_positions_real = spatializer.speaker_positions
    speaker_positions_screen = _real_to_screen_coords(speaker_positions_real)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Convert current mouse position to real coordinates for display
        current_real_pos = _screen_to_real_coords(mouse_x, mouse_y)
        
        # Draw everything
        screen.fill(BLACK)
        
        # Draw speakers
        for i, pos in enumerate(speaker_positions_screen):
            pygame.draw.circle(screen, BLUE, pos, SPEAKER_RADIUS)
            # Add speaker number
            font = pygame.font.Font(None, 24)
            text = font.render(str(i + 1), True, WHITE)
            text_rect = text.get_rect(center=(pos[0], pos[1] - SPEAKER_RADIUS - 15))
            screen.blit(text, text_rect)
        
        # Draw sound position (where mouse is pointing)
        sound_screen_pos = _real_to_screen_coords([current_real_pos])[0]
        pygame.draw.circle(screen, RED, sound_screen_pos, SOUND_RADIUS)
        
        # Draw crosshair at mouse position
        pygame.draw.line(screen, YELLOW, (mouse_x - 10, mouse_y), (mouse_x + 10, mouse_y), 2)
        pygame.draw.line(screen, YELLOW, (mouse_x, mouse_y - 10), (mouse_x, mouse_y + 10), 2)
        
        # Draw instructions
        font = pygame.font.Font(None, 36)
        instructions = [
            "Move mouse to position sine wave (440 Hz)",
            "Blue circles = Speakers",
            "Red circle = Sound position",
            "Press ESC or close window to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, WHITE)
            screen.blit(text, (10, 10 + i * 30))
        
        # Show current position
        pos_text = f"Position: ({current_real_pos[0]:.2f}, {current_real_pos[1]:.2f})"
        pos_surface = font.render(pos_text, True, GREEN)
        screen.blit(pos_surface, (10, WINDOW_HEIGHT - 40))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    # Generate a simple sine wave (exactly like circular_sine.py)
    sample_rate = get_sample_rate()
    duration = 30  # Duration in seconds
    frequency = 440  # A4 note frequency in Hz
    
    # Create a sine wave
    sample_count = int(sample_rate * duration)
    t = np.linspace(0, duration, sample_count, False)
    # Amplitude 0.5 to avoid clipping (same as circular_sine.py)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Create a playback sound object (starting at center)
    sound_object = SO_Playback(sine_wave)
    
    # Set up the spatial audio scene
    spatializer = Spatializer()
    scene = Scene(spatializer)
    scene.register(sound_object)
    
    # Start pygame thread
    pygame_thread_handle = threading.Thread(target=pygame_thread, daemon=True)
    pygame_thread_handle.start()
    
    print("Interactive sine wave positioning started!")
    print("Move your mouse around the window to control the sound position.")
    
    # Audio processing (exactly like circular_sine.py structure)
    sound_streamer = SoundNetworkStreamer()
    
    # Implement precise real-time timing
    chunk_duration = CHUNKSIZE / SAMPLING_RATE
    timing_start = time.perf_counter()
    
    try:
        for j, chunk in enumerate(scene.run()):
            if not running:
                break
                
            # Get current mouse position and convert to real coordinates
            real_position = _screen_to_real_coords(mouse_x, mouse_y)
            
            # Update sound object position (exactly like circular_sine.py)
            sound_object.set_position(np.array([real_position[0], real_position[1]], dtype=float))
            
            # Print position information (optional, like circular_sine.py)
            if j % 20 == 0:  # Print every 20th update to reduce console output
                print(f"Chunk: {j}, Position: ({real_position[0]:.2f}, {real_position[1]:.2f})")
            
            # Send audio to output (exactly like circular_sine.py)
            chunk = np.clip(chunk, -1, 1)
            sound_streamer.send(chunk)
            
            # Schedule next chunk send time (precise timing, exactly like circular_sine.py)
            next_time = timing_start + (j + 1) * chunk_duration
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        running = False
        pygame_thread_handle.join(timeout=1)
        print("Application closed.")