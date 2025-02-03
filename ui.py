import streamlit as st
from PIL import Image
import cv2
import os
from SOURCE.yolo_files import detect
from SOURCE.gan_files import test
from SOURCE.vgg_finetuned_model import vgg_verify
from helper_fns import gan_utils
import shutil
import glob
import SessionState

MEDIA_ROOT = 'media/documents/'
SIGNATURE_ROOT = 'media/UserSignaturesSquare/'
YOLO_RESULT = 'results/yolov5/'
YOLO_OP = 'crops/DLSignature/'
GAN_IPS = 'results/gan/gan_signdata_kaggle/gan_ips/testB'
GAN_OP = 'results/gan/gan_signdata_kaggle/test_latest/images/'
GAN_OP_RESIZED = 'results/gan/gan_signdata_kaggle/test_latest/images/'


def select_cleaned_image(selection):
    return GAN_OP + selection + '_fake.png'

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def signature_verify(selection):
    anchor_image = SIGNATURE_ROOT + selection + '.png'
    feature_set = vgg_verify.verify(anchor_image, GAN_OP_RESIZED)
    for image, score in feature_set:
        columns = [column for column in st.columns(3)]
        columns[0].image(anchor_image)
        columns[1].image(image)
        columns[2].write(score)

def signature_cleaning(selection, yolo_op):
    copy_and_overwrite(yolo_op, GAN_IPS)
    test.clean()

    cleaned_image = select_cleaned_image(selection)
    st.image(cleaned_image)

def signature_detection(selection):
    detect.detect(MEDIA_ROOT)
    latest_detection = max(glob.glob(os.path.join(YOLO_RESULT, '*/')), key=os.path.getmtime)
    print(os.path.join(latest_detection, YOLO_OP))
    gan_utils.resize_images(os.path.join(latest_detection, YOLO_OP).replace("\\","/"))
    selection_detection =latest_detection + YOLO_OP + selection + '.jpg'
    st.image(selection_detection)
    return latest_detection + YOLO_OP

def select_document():
    left, right = st.columns(2)
    selection = str(left.selectbox('Select document to run inference', [1, 2]))
    selection_image = MEDIA_ROOT+selection+'.png'
    right.image(selection_image, use_column_width='always')
    return selection

def main():
    session_state = SessionState.get(
        selection = '',
        yolo_op = '',
        detect_button = False,
        clean_button = False,
        verify_button = False,
        
    )
    st.write('Revolutionizing Document Security: A Comprehensive Deep Learning Approach For Signature Detection And Verification')

    session_state.selection = select_document()
    
    detect_button = st.button('Detect Signature')
    if detect_button:
        session_state.detect_button = True
    if session_state.detect_button:
        session_state.yolo_op = signature_detection(session_state.selection)
        
        clean_button = st.button('Clean Signature')
        if clean_button:
            session_state.clean_button = True
        if session_state.clean_button:
            signature_cleaning(session_state.selection, session_state.yolo_op)
        
            verify_button = st.button('Verify Signature')
            if verify_button:
                session_state.verify_button = True
            if session_state.verify_button:
                signature_verify(session_state.selection)

main()