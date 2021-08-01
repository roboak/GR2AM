import time

from flask import Flask, Response, redirect, render_template, request, session, url_for
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
from pynput.keyboard import Key,Controller
import math


app = Flask(__name__)
@app.route("/volume_change/<flag>", methods=['GET'])
def volume_up(flag):
    keyboard = Controller()
    if flag == 'down':
        for i in range(5):
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
    if flag == 'up':
        for i in range(5):
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
    return Response("Volume {}".format(flag))

@app.route("/play_pause", methods=['GET'])
def toggle_play_pause():
    keyboard = Controller()
    keyboard.press(Key.media_play_pause)
    keyboard.release(Key.media_play_pause)
    return Response("Ok")

@app.route("/media_change/<flag>", methods=['GET'])
def toggle_play_pause(flag):
    keyboard = Controller()
    if(flag == 'next'):
        keyboard.press(Key.media_next)
        keyboard.release(Key.media_next)
    if(flag == 'prev'):
        keyboard.press(Key.media_previous)
        keyboard.release(Key.media_previous)
    return Response("Ok")

@app.route("/brightness_change/<flag>", methods=['GET', 'POST'])
def brightness_change(flag):
    # get current brightness value
    current_brightness = sbc.get_brightness()
    print(current_brightness)
    if flag == 'up':
        new_brightness = min(100, int(1.25*current_brightness))
    if flag == 'down':
        new_brightness = max(0,int(0.75*current_brightness))
    sbc.set_brightness(new_brightness)
    print(sbc.get_brightness())
    return Response("All Ok")



if __name__ == "__main__":
    app.run(debug=True, port=8081)
