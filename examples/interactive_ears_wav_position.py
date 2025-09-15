import pygame
import numpy as np
import time
import threading
import soundfile as sf
from spatial_audio_ai import (
    SO_Playback, 
    Scene,
    SoundNetworkStreamer,
    get_sample_rate
)
from spatial_audio_ai.tools.spatializer import EarsSpatializer, CHUNKSIZE, SAMPLING_RATE

# Initialize pygame
pygame.init()

# Window settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
SPEAKER_RADIUS = 8
SOUND_RADIUS = 12
EARS_RADIUS = 10

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
ORANGE = (255, 165, 0)

# Global variables for mouse position sharing between threads
sound_x, sound_y = 0, 0
ears_x, ears_y = 0, 0
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
    global sound_x, sound_y, ears_x, ears_y, running
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Interactive Ears+WAV Positioning - Left click: sound, Right click: ears")
    clock = pygame.time.Clock()
    
    # Get speaker positions for visualization
    spatializer = EarsSpatializer()
    speaker_positions_real = spatializer.speaker_positions
    speaker_positions_screen = _real_to_screen_coords(speaker_positions_real)
    
    # Initialize positions
    sound_x, sound_y = WINDOW_WIDTH // 2 + 100, WINDOW_HEIGHT // 2
    ears_x, ears_y = WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if event.button == 1:  # Left click - control sound position
                    sound_x, sound_y = mouse_x, mouse_y
                elif event.button == 3:  # Right click - control ears position
                    ears_x, ears_y = mouse_x, mouse_y
        
        # Convert current positions to real coordinates for display
        current_sound_pos = _screen_to_real_coords(sound_x, sound_y)
        current_ears_pos = _screen_to_real_coords(ears_x, ears_y)
        
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
        
        # Draw connection line from ears to sound
        pygame.draw.line(screen, GRAY, (ears_x, ears_y), (sound_x, sound_y), 2)
        
        # Extend the line beyond the sound to show direction
        if abs(sound_x - ears_x) > 1 or abs(sound_y - ears_y) > 1:
            # Calculate extended line
            dx = sound_x - ears_x
            dy = sound_y - ears_y
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                # Extend the line by 200 pixels
                extend_factor = 200 / length
                end_x = sound_x + dx * extend_factor
                end_y = sound_y + dy * extend_factor
                # Clamp to screen bounds
                end_x = max(0, min(WINDOW_WIDTH, end_x))
                end_y = max(0, min(WINDOW_HEIGHT, end_y))
                pygame.draw.line(screen, YELLOW, (sound_x, sound_y), (end_x, end_y), 1)
        
        # Draw sound position (where file is playing)
        pygame.draw.circle(screen, PURPLE, (sound_x, sound_y), SOUND_RADIUS)
        
        # Draw ears position
        pygame.draw.circle(screen, GREEN, (ears_x, ears_y), EARS_RADIUS)
        
        # Draw instructions
        font = pygame.font.Font(None, 36)
        instructions = [
            "Left click to position monster.wav",
            "Right click to position ears",
            "Blue circles = Speakers",
            "Purple circle = Sound position",
            "Green circle = Ears position",
            "Gray line = Ears to sound vector",
            "Yellow line = Extended direction",
            "Press ESC or close window to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, WHITE)
            screen.blit(text, (10, 10 + i * 30))
        
        # Show current positions
        pos_text = f"Sound: ({current_sound_pos[0]:.2f}, {current_sound_pos[1]:.2f})"
        ears_text = f"Ears: ({current_ears_pos[0]:.2f}, {current_ears_pos[1]:.2f})"
        distance = np.linalg.norm(current_sound_pos - current_ears_pos)
        distance_text = f"Distance: {distance:.2f}"
        
        pos_surface = font.render(pos_text, True, PURPLE)
        ears_surface = font.render(ears_text, True, GREEN)
        distance_surface = font.render(distance_text, True, YELLOW)
        
        screen.blit(pos_surface, (10, WINDOW_HEIGHT - 100))
        screen.blit(ears_surface, (10, WINDOW_HEIGHT - 70))
        screen.blit(distance_surface, (10, WINDOW_HEIGHT - 40))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

def load_and_prepare_wav(file_path):
    """Load WAV file and prepare it for continuous playback"""
    try:
        # Load the WAV file
        audio_data, original_sample_rate = sf.read(file_path)
        print(f"Loaded {file_path}")
        print(f"Original sample rate: {original_sample_rate} Hz")
        print(f"Audio shape: {audio_data.shape}")
        print(f"Duration: {len(audio_data) / original_sample_rate:.2f} seconds")
        
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            print("Converted stereo to mono")
        
        # Resample if necessary
        target_sample_rate = get_sample_rate()
        if original_sample_rate != target_sample_rate:
            # Simple resampling (basic interpolation)
            duration = len(audio_data) / original_sample_rate
            new_length = int(duration * target_sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data) - 1, new_length),
                np.arange(len(audio_data)),
                audio_data
            )
            print(f"Resampled from {original_sample_rate} Hz to {target_sample_rate} Hz")
        
        # Normalize amplitude to prevent clipping
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0:
            audio_data = audio_data / max_amplitude * 0.5  # Scale to 0.5 max amplitude
        
        # Create a longer looping version for continuous playback
        # Repeat the audio multiple times to create a long buffer
        loop_count = max(10, int(60 * target_sample_rate / len(audio_data)))  # At least 60 seconds
        looped_audio = np.tile(audio_data, loop_count)
        
        print(f"Created looped version with {loop_count} repetitions")
        print(f"Total duration: {len(looped_audio) / target_sample_rate:.2f} seconds")
        
        return looped_audio
        
    except Exception as e:
        print(f"Error loading WAV file: {e}")
        print("Falling back to sine wave...")
        # Fallback to sine wave if file loading fails
        sample_rate = get_sample_rate()
        duration = 30
        frequency = 440
        sample_count = int(sample_rate * duration)
        t = np.linspace(0, duration, sample_count, False)
        return 0.5 * np.sin(2 * np.pi * frequency * t)

if __name__ == "__main__":
    # Load the WAV file (exactly like original interactive_wav_position.py)
    wav_file_path = "examples/monster.wav"
    print(f"Loading WAV file: {wav_file_path}")
    audio_data = load_and_prepare_wav(wav_file_path)
    
    # Create a playback sound object (starting at center)
    sound_object = SO_Playback(audio_data)
    
    # Set up the spatial audio scene with EarsSpatializer
    spatializer = EarsSpatializer(ears_position=np.array([0.0, 0.0]))
    scene = Scene(spatializer)
    scene.register(sound_object)
    
    # Start pygame thread
    pygame_thread_handle = threading.Thread(target=pygame_thread, daemon=True)
    pygame_thread_handle.start()
    
    print("Interactive ears+WAV positioning started!")
    print("Left click to control sound position, right click to control ears position.")
    
    # Audio processing (exactly like interactive_wav_position.py structure)
    sound_streamer = SoundNetworkStreamer()
    
    # Implement precise real-time timing
    chunk_duration = CHUNKSIZE / SAMPLING_RATE
    timing_start = time.perf_counter()
    
    try:
        for j, chunk in enumerate(scene.run()):
            if not running:
                break
                
            # Get current positions and convert to real coordinates
            sound_real_position = _screen_to_real_coords(sound_x, sound_y)
            ears_real_position = _screen_to_real_coords(ears_x, ears_y)
            
            # Update sound object position (exactly like interactive_wav_position.py)
            sound_object.set_position(np.array([sound_real_position[0], sound_real_position[1]], dtype=float))
            
            # Update ears position in spatializer
            spatializer.set_ears_position(np.array([ears_real_position[0], ears_real_position[1]], dtype=float))
            
            # Print position information (optional, like interactive_wav_position.py)
            if j % 100 == 0:  # Print every 100th update to reduce console output
                print(f"Chunk: {j}, Sound: ({sound_real_position[0]:.2f}, {sound_real_position[1]:.2f}), Ears: ({ears_real_position[0]:.2f}, {ears_real_position[1]:.2f})")
            
            # Send audio to output (exactly like interactive_wav_position.py)
            chunk = np.clip(chunk, -1, 1)
            sound_streamer.send(chunk)
            
            # Schedule next chunk send time (precise timing, exactly like interactive_wav_position.py)
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
