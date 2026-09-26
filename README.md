Vantage — 3D Car Customizer with Webcam Hand-Gesture Control

Vantage is a desktop app that renders a fully customizable car in real time using
PyOpenGL, complete with a build-up animation (dots -> wireframe -> solid), a small
avatar standing beside it, and a simple road environment. You can control it with
your mouse and keyboard, or — if you have a webcam — with hand gestures, powered by
OpenCV + MediaPipe.

--------------------------------------------------------------------------------
FEATURES
--------------------------------------------------------------------------------
- Real-time 3D car rendering (raw OpenGL immediate mode) with a progressive
  "self-assembly" intro animation
- Full customization: paint color, wheels, headlights/taillights, spoiler, and
  interior trim, each affecting a running total price
- A small road scene with lamp posts, curbs, and lane markings for atmosphere
- Save/load your configuration to disk
- Mouse + keyboard controls that always work, with optional webcam hand-gesture
  control layered on top

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
- Python 3.9+
- A webcam (optional — only needed for hand-gesture control; everything else
  works fine without one)

Install dependencies:

    pip install pygame PyOpenGL PyOpenGL_accelerate opencv-python mediapipe numpy

--------------------------------------------------------------------------------
RUNNING IT
--------------------------------------------------------------------------------
    python car.py

The app opens in a single window. If a webcam is detected, gesture control is
available but starts OFF — press 'g' to enable it.

--------------------------------------------------------------------------------
CONTROLS
--------------------------------------------------------------------------------

Mouse & keyboard:

    Mouse drag             Orbit the camera
    Scroll wheel           Zoom the camera
    Left / Right arrows    Rotate the car
    Up / Down arrows       Tilt the camera
    1-7                    Pick a paint color
    w                      Cycle wheels
    l                      Cycle headlights/taillights
    p                      Cycle spoiler
    i                      Cycle interior
    r                      Replay the build-up animation
    Space                  Skip straight to the finished car
    g                      Toggle hand-gesture control
    s                      Save the current configuration
    Esc / q                Quit

Hand gestures (once enabled with 'g'):

    Open hand, swipe left/right           Rotate the car a step
    Open hand, twist your wrist           Rotate the car smoothly
    Pinch (thumb + index), move up/down   Zoom the camera in/out
    Hold up 1-4 fingers                   Jump straight to that paint color (1st-4th swatch)
    Closed fist                           Pause / resume the animation

A small webcam preview window shows your hand tracking and the currently
recognized gesture. Press 'q' inside that window (or 'g' in the main window) to
turn gestures off again.

--------------------------------------------------------------------------------
SAVED CONFIGURATION
--------------------------------------------------------------------------------
Pressing 's' writes your current setup to vantage_config.json in the folder
you launched the app from. On the next launch, that file is loaded
automatically if present, so your car reopens exactly as you left it.

Example:

    {
      "color": 2,
      "wheel": 1,
      "light": 0,
      "spoiler": 2,
      "interior": 1,
      "price": 38900
    }

--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------
.
├── car.py                Everything: rendering, gestures, app loop
└── vantage_config.json   Created after your first save (not tracked in git)

Consider adding a .gitignore entry for vantage_config.json if you don't want
your personal saved setup committed to the repo:

    vantage_config.json
