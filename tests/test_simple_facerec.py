import cv2
from simple_facerec import SimpleFacerec
import sys
import pytest


def test_load_encoding_images(capsys):
    
    SimpleFacerec().load_encoding_images("C:\Users\ichrak\Desktop\facialRecongnition\images")
    out, err = capsys.readouterr()
    assert out == "4 encoding images found"
    assert out == "Encoding images loaded"
    assert out == "loaded"
    assert out == " 7 Encoding images loaded"


def test_detect_known_faces():

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    faceLocation = 176
    faceName = 'Aroussi teyeb'
    faceLocation1 = 215
    faceName1 = 'mounir'
    faceLocation3 = 215
    faceName3 = 'souhaila'

    assert SimpleFacerec().detect_known_faces(frame) == faceLocation, faceName
    assert SimpleFacerec().detect_known_faces(frame) == faceLocation1, faceName1
    assert SimpleFacerec().detect_known_faces(frame) == faceLocation3, faceName3


def known_faces_errors():
    with pytest.raises(ValueError, match="line 16, in <module>face_locations, face_names = sfr.detect_known_faces(frame)"):
        with pytest.raises(ValueError, match="line 48, in detect_known_facesface_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)"):

            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            faceLocation = 176
            faceName = 'Aroussi teyeb'
            SimpleFacerec().detect_known_faces(frame) == faceLocation, faceName
