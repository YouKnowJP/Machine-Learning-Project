import numpy as np
import time
from mss import mss
from PIL import Image, ImageOps
from PIL.Image import Resampling
from collections import deque
from pynput.keyboard import Controller, Key
import config

# Create a global keyboard controller instance
keyboard_controller = Controller()

class Environment:
    def __init__(self):
        self.mon = config.MONITOR
        self.sct = mss()
        self.image_bank = deque(maxlen=4)
        self.action_memory = 2  # Initial action (no key pressed)
        self.start_time = None

    @staticmethod
    def start_game(countdown=3):
        """
        Starts the game after a countdown. Does not rely on any instance attributes,
        so we mark it as static to remove PyCharm's warning.
        """
        for i in reversed(range(countdown)):
            print("Game starting in", i)
            time.sleep(1)

    def reset(self):
        self.start_time = time.time()
        # Simulate a key press to restart the game using pynput
        keyboard_controller.press(Key.space)
        time.sleep(0.5)
        keyboard_controller.release(Key.space)
        self.image_bank.clear()
        state, reward, done = self.step(0)
        return state

    def step(self, action):
        actions = {0: Key.space, 1: Key.down}
        # If the action has changed, release the previous key
        if action != self.action_memory:
            if self.action_memory in actions:
                keyboard_controller.release(actions[self.action_memory])
            if action in actions:
                keyboard_controller.press(actions[action])
        self.action_memory = action

        screenshot = self.sct.grab(self.mon)
        img = np.array(screenshot)[:, :, 0]  # Use the first channel (grayscale)
        processed_img = self._process_img(img)
        state = self._update_image_bank(processed_img)
        done = Environment._is_done(processed_img)
        reward = Environment._get_reward(done)
        return state, reward, done

    def _process_img(self, img):
        """
        Process a raw screenshot (grayscale array) into a normalized float32 array.
        """
        pil_img = Image.fromarray(img)
        # Always use Resampling.LANCZOS (Pillow 9.1+). Remove ANTIALIAS fallback.
        pil_img = pil_img.resize((384, 76), Resampling.LANCZOS)

        # Invert image if it's bright (night mode detection, etc.)
        if np.sum(np.array(pil_img)) > 2_000_000:
            pil_img = ImageOps.invert(pil_img)

        # Adjust contrast
        pil_img = Environment._adjust_contrast(pil_img)

        # Convert to float32 and normalize
        img_array = np.array(pil_img, dtype=np.float32)
        img_array = img_array / 255.0
        return img_array

    @staticmethod
    def _adjust_contrast(img):
        """
        Adjust contrast by clipping pixel values and scaling.
        No instance attributes used -> static method.
        """
        min_val = 32
        max_val = 171
        img_array = np.array(img, dtype=np.float32)
        img_array = np.clip(img_array, min_val, max_val)
        img_array = (img_array - min_val) / (max_val - min_val)
        return Image.fromarray((img_array * 255).astype(np.uint8))

    def _update_image_bank(self, img):
        """
        Maintain a rolling window (deque) of the last 4 frames.
        This method uses self.image_bank -> must remain an instance method.
        """
        if len(self.image_bank) < 4:
            for _ in range(4 - len(self.image_bank)):
                self.image_bank.append(img)
        else:
            self.image_bank.popleft()
            self.image_bank.append(img)
        state = np.stack(self.image_bank, axis=-1)
        return state

    @staticmethod
    def _get_reward(done):
        """
        Return a reward of -15 if done, otherwise +1.
        No instance attributes used -> static method.
        """
        return -15 if done else 1

    @staticmethod
    def _is_done(img):
        """
        Check a specific region of the image to determine if the game is over.
        No instance attributes used -> static method.
        """
        region = img[30:50, 180:203]
        val = np.sum(region)
        expected_val_day = 243.53
        expected_val_night = 331.94
        if abs(val - expected_val_day) > 15 and abs(val - expected_val_night) > 15:
            return False
        return True
