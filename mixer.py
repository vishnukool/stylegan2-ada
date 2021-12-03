import requests
from urllib import request, parse
import io
import dnnlib
import dnnlib.tflib as tflib
import pickle
import numpy as np
from projector import Projector
from PIL import Image
import re
import base64
from io import BytesIO
import boto3
import uuid

img_res = 256


def find_nearest(arr, val):
    "Element in nd array `arr` closest to the scalar value `a0`"
    idx = np.abs(arr - val).argmin()
    return arr.flat[idx]


def url_to_b64(url):
    return base64.b64encode(requests.get(url).content)


def url_to_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    white_bg_image = Image.new("RGBA", img.size, "WHITE")  # Create a white rgba background
    white_bg_image.paste(img, (0, 0), img)
    white_bg_image = resizeimage.resize_contain(white_bg_image, [256, 256])
    white_bg_image = white_bg_image.convert('RGB')
    return white_bg_image


def base64_to_image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img


def pil_image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def resize(image_pil, width, height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_pil = image_pil.convert('RGBA')
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGB', (width, height), "WHITE")
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset, image_resize)
    return background.convert('RGB')


def cropAndRezieBase64Img(b64Img):
    if (isinstance(b64Img, str)):
        temp = base64_to_image(b64Img)
    else:
        temp = base64_to_image(b64Img.decode("utf-8"))
    data = parse.urlencode(
        {'image_file_b64': b64Img, 'crop': 'true', 'crop_margin': '10px', 'type': 'product',
         'format': 'jpg'}).encode()
    req = request.Request('https://api.remove.bg/v1.0/removebg', data=data)  # this will make the method "POST"
    req.add_header('X-Api-Key', 'hhm7D13ckriCprFLHLXyaaiP')
    response = request.urlopen(req).read()
    baseImage = Image.open(io.BytesIO(response))
    baseImage = resize(baseImage, img_res, img_res)
    img = pil_image_to_base64(baseImage)
    return img


def mixLatents(network_pkl, model_name, imageId, inputs):
    thetaAngles = [40, 30, 20, 10, 0, 350, 340, 330, 320];
    widthInches = np.array([52, 56, 60, 64, 68, 72, 74, 78, 82, 86, 90]);
    lengthInches = np.array([36]);
    heightInches = np.array([32]);
    styleB64img = inputs[0]
    widthInches = find_nearest(widthInches, inputs[1])
    lengthInches = find_nearest(lengthInches, inputs[2])
    heightInches = find_nearest(heightInches, inputs[3])
    angle = 15

    proj = Projector()
    angleLatentPath = 'out/base'
    styleLatentPath = 'out/style'

    widthInMeter = widthInches * 0.0254
    depthInMeter = lengthInches * 0.0254
    heightInMeter = heightInches * 0.0254

    aws_key = 'SOME_KEY'
    aws_secret = 'SOME_SECRET'

    session = boto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )
    s3 = boto3.client('s3', aws_access_key_id=aws_key,
                      aws_secret_access_key=aws_secret)
    s3Resource = session.resource('s3')

    # baseB64img = cropAndRezieBase64Img(url_to_b64(baseUrl))
    styleB64img = cropAndRezieBase64Img(styleB64img)
    # proj.project2('network-snapshot-009800.pkl', baseB64img, angleLatentPath, False, 1)
    proj.project2('network-snapshot-009800.pkl', styleB64img, styleLatentPath, False, 1)

    Gs_syn_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False,
        'minibatch_size': 4
    }
    tflib.init_tf()
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    # model_name = 'armless_sofa'
    rotatedImagesB64 = []

    for i, angle in enumerate(thetaAngles):
        i += 1
        print(f'{i}/{len(thetaAngles)}')
        baseLatent = f'{model_name}_{angle}_{widthInches}_{lengthInches}_{heightInches}_dlatents.npz'
        # baseLatent = 'armless_sofa_0_56_36_32_dlatents.npz'
        s3.download_file('sofa-latents', baseLatent,
                         baseLatent)
        with np.load(baseLatent) as latent1:
            with np.load(styleLatentPath + '/dlatents.npz') as latent2:
                lat1 = latent1['dlatents']
                lat2 = latent2['dlatents']
                col_styles = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                # col_styles = [10,11,12,13,14,15,16,17]
                # col_styles = [10,11,12,13]
                lat1[0][col_styles] = lat2[0][col_styles]
                image = Gs.components.synthesis.run(lat1, **Gs_syn_kwargs)[0]
                mixImg = Image.fromarray(image, 'RGB')
                respB64 = pil_image_to_base64(mixImg)
                rotatedImagesB64.append(respB64)
                s3ImageName = imageId + f'-{i:03}' + '.jpg'
                print(s3ImageName)
                obj = s3Resource.Object('homely-demo-renders', s3ImageName)
                obj.put(Body=base64.b64decode(respB64))
    print(imageId)
    return {'id': imageId}