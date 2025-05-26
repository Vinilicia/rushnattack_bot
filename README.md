# Rush'n Attack (NES) Game Bot

This project is a simple bot implementation for the **Rush'n Attack** game on the **NES** console, using the `gym-retro` library.

## Requirements

- Python **3.7**
- `gym-retro` **0.7.0**
- `opencv-python` (cv2)

## Installation

Use `pyenv` to ensure the correct Python version, and set up the environment:
```bash
pyenv install 3.7.17
pyenv virtualenv 3.7.17 myenv37
pyenv activate myenv37
pip install gym-retro==0.7.0 opencv-python
```

## Importing the ROM

```bash
python -m retro.import .
```

## Running the Bot

```bash
python rush_n_attack_bot.py
```
