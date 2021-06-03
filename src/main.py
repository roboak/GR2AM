from gesture_capturing import GestureCapture
from os.path import dirname, abspath
from pathlib import Path
from meta_data_class import GestureMetaData

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parent_directory = dirname(dirname(abspath(__file__)))
    print(parent_directory)
    parent_directory = Path(parent_directory)
    path = parent_directory / "HandDataset"
    gestureMetaData = GestureMetaData(gesture_name="gesture22")
    gesture = GestureCapture(folder_location=str(path), gesture_meta_data=gestureMetaData, camera_input_value=0)
    gesture.get_frame()
