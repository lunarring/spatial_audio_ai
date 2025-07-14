#!/usr/bin/env python3
"""
Spatial Sine Visual - Interactive 2D visualization of spatial audio with pygame

This script provides a real-time visualization of a sine wave sound source that can be
moved around in 2D space. Shows speakers, sound source position, and includes frequency control.
"""

import pygame
import numpy as np
import threading
import time
import sys
import math
from typing import Tuple, Optional

# Import spatial audio components
from spatial_audio_ai.tools.sound_system import SoundSystem
from spatial_audio_ai.tools.sound_objects import SO_PlaybackSine
from spatial_audio_ai.tools.spatializer import Spatializer
from spatial_audio_ai.config import N_SPEAKERS, SAMPLING_RATE

class SpatialSineVisualizer:
    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height
        self.world_size = 10.0  # 10x10 meter space
        self.running = False
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        self.YELLOW = (255, 255, 0)
        
        # Audio components
        self.sound_system = None
        self.spatializer = None
        self.sine_object = None
        
        # Visualization state
        self.sound_position = np.array([0.0, 0.0], dtype=float)
        self.frequency = 440.0
        self.dragging = False
        
        # Speaker positions (arranged in a circle for N_SPEAKERS)
        self.speaker_positions = self._generate_speaker_positions()
        
        # Pygame elements
        self.screen = None
        self.clock = None
        self.font = None
        
    def _generate_speaker_positions(self) -> list:
        """Generate speaker positions around the room perimeter"""
        positions = []
        radius = self.world_size * 0.4  # Place speakers around 80% of the room
        center_x, center_y = self.world_size / 2, self.world_size / 2
        
        for i in range(N_SPEAKERS):
            angle = 2 * math.pi * i / N_SPEAKERS
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((x, y))
        
        return positions
    
    def world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        x, y = world_pos
        screen_x = int((x / self.world_size) * self.width)
        screen_y = int((1 - y / self.world_size) * self.height)  # Flip Y axis
        return screen_x, screen_y
    
    def screen_to_world(self, screen_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        x, y = screen_pos
        world_x = (x / self.width) * self.world_size
        world_y = (1 - y / self.height) * self.world_size  # Flip Y axis
        return world_x, world_y
    
    def initialize_pygame(self):
        """Initialize pygame components"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Spatial Sine Visualizer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
    def initialize_audio(self):
        """Initialize audio system"""
        try:
            self.sound_system = SoundSystem(mock_mode=False)
            self.spatializer = Spatializer(self.sound_system)
            
            # Create sine sound object
            self.sine_object = SO_PlaybackSine(
                frequency=self.frequency,
                amplitude=0.3,
                position=self.sound_position.copy()
            )
            
            # Add to spatializer
            self.spatializer.add_sound_object(self.sine_object)
            
            print(f"Audio initialized with {N_SPEAKERS} speakers")
            return True
            
        except Exception as e:
            print(f"Audio initialization failed: {e}")
            print("Running in visual-only mode")
            return False
    
    def draw_speakers(self):
        """Draw speaker positions"""
        for i, (x, y) in enumerate(self.speaker_positions):
            screen_x, screen_y = self.world_to_screen((x, y))
            pygame.draw.circle(self.screen, self.BLUE, (screen_x, screen_y), 8)
            
            # Draw speaker label
            label = self.font.render(f"S{i+1}", True, self.BLUE)
            self.screen.blit(label, (screen_x - 10, screen_y - 25))
    
    def draw_sound_source(self):
        """Draw the sine wave sound source"""
        screen_x, screen_y = self.world_to_screen(self.sound_position)
        
        # Draw source with pulsing effect based on frequency
        pulse = abs(math.sin(time.time() * self.frequency / 100)) * 5 + 10
        pygame.draw.circle(self.screen, self.RED, (screen_x, screen_y), int(pulse))
        
        # Draw center dot
        pygame.draw.circle(self.screen, self.WHITE, (screen_x, screen_y), 3)
    
    def draw_frequency_slider(self):
        """Draw frequency control slider"""
        slider_x = 50
        slider_y = self.height - 80
        slider_width = 300
        slider_height = 20
        
        # Slider background
        pygame.draw.rect(self.screen, self.GRAY, 
                        (slider_x, slider_y, slider_width, slider_height))
        
        # Slider handle position (20Hz to 2000Hz range)
        min_freq, max_freq = 20, 2000
        handle_pos = ((self.frequency - min_freq) / (max_freq - min_freq)) * slider_width
        handle_x = slider_x + int(handle_pos)
        
        # Draw handle
        pygame.draw.circle(self.screen, self.YELLOW, 
                          (handle_x, slider_y + slider_height // 2), 12)
        
        # Labels
        freq_label = self.font.render(f"Frequency: {self.frequency:.1f} Hz", True, self.WHITE)
        self.screen.blit(freq_label, (slider_x, slider_y - 30))
    
    def draw_info_panel(self):
        """Draw information panel"""
        info_lines = [
            f"Position: ({self.sound_position[0]:.2f}, {self.sound_position[1]:.2f})",
            f"Speakers: {N_SPEAKERS}",
            f"Sample Rate: {SAMPLING_RATE} Hz",
            "",
            "Controls:",
            "• Drag red circle to move sound",
            "• Drag yellow handle to change frequency",
            "• ESC to quit"
        ]
        
        for i, line in enumerate(info_lines):
            color = self.LIGHT_GRAY if line.startswith("•") else self.WHITE
            text = self.font.render(line, True, color)
            self.screen.blit(text, (50, 50 + i * 25))
    
    def handle_mouse_input(self, mouse_pos: Tuple[int, int], mouse_pressed: bool):
        """Handle mouse input for dragging sound source and frequency slider"""
        # Check frequency slider
        slider_x = 50
        slider_y = self.height - 80
        slider_width = 300
        slider_height = 20
        
        if (slider_x <= mouse_pos[0] <= slider_x + slider_width and
            slider_y <= mouse_pos[1] <= slider_y + slider_height):
            if mouse_pressed:
                # Update frequency based on slider position
                relative_pos = (mouse_pos[0] - slider_x) / slider_width
                relative_pos = max(0, min(1, relative_pos))
                
                min_freq, max_freq = 20, 2000
                self.frequency = min_freq + relative_pos * (max_freq - min_freq)
                
                if self.sine_object:
                    self.sine_object.set_frequency(self.frequency)
                return
        
        # Check sound source dragging
        if mouse_pressed:
            world_pos = self.screen_to_world(mouse_pos)
            # Clamp to world bounds
            world_x = max(0, min(self.world_size, world_pos[0]))
            world_y = max(0, min(self.world_size, world_pos[1]))
            
            self.sound_position = np.array([world_x, world_y], dtype=float)
            
            if self.sine_object:
                self.sine_object.set_position(self.sound_position.copy())
    
    def run(self):
        """Main visualization loop"""
        self.initialize_pygame()
        audio_available = self.initialize_audio()
        
        self.running = True
        print("Spatial Sine Visualizer started")
        print("Drag the red circle to move the sound source")
        print("Drag the yellow slider to change frequency")
        
        try:
            while self.running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                
                # Handle mouse input
                mouse_pos = pygame.mouse.get_pos()
                mouse_pressed = pygame.mouse.get_pressed()[0]
                self.handle_mouse_input(mouse_pos, mouse_pressed)
                
                # Clear screen
                self.screen.fill(self.BLACK)
                
                # Draw grid
                for i in range(0, int(self.world_size) + 1):
                    screen_pos = self.world_to_screen((i, 0))
                    pygame.draw.line(self.screen, (30, 30, 30), 
                                   (screen_pos[0], 0), (screen_pos[0], self.height))
                    
                    screen_pos = self.world_to_screen((0, i))
                    pygame.draw.line(self.screen, (30, 30, 30),
                                   (0, screen_pos[1]), (self.width, screen_pos[1]))
                
                # Draw components
                self.draw_speakers()
                self.draw_sound_source()
                self.draw_frequency_slider()
                self.draw_info_panel()
                
                # Update display
                pygame.display.flip()
                self.clock.tick(60)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.spatializer:
            try:
                self.spatializer.stop()
            except:
                pass
                
        if self.sound_system:
            try:
                # Sound system cleanup handled by spatializer
                pass
            except:
                pass
        
        pygame.quit()
        print("Spatial Sine Visualizer stopped")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Spatial Sine Visualizer")
        print("Interactive 2D visualization of spatial sine audio")
        print("Usage: python spatial_sine_visual.py")
        return
    
    visualizer = SpatialSineVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main() 