#%%
import os
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
import skimage
from skimage.transform import PiecewiseAffineTransform, warp
import PIL
import skimage
from skimage.transform import PiecewiseAffineTransform, warp
import mediapipe as mp
import os,argparse,uuid
import dlib,cv2,filetype
import numpy as np
from imutils import face_utils
import config
import webcolors, filetype
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image, ImageDraw
import tempfile
import os
import tempfile
from vtkplotter import load, show
import sys

def imshow(img):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    return ax

def load_obj(obj_filename):
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = np.array(verts)
    uvcoords = np.array(uvcoords)
    faces = np.array(faces); faces = faces.reshape(-1, 3) - 1
    uv_faces = np.array(uv_faces); uv_faces = uv_faces.reshape(-1, 3) - 1
    
    return (
        verts,
        uvcoords,
        faces,
        uv_faces
    )

def write_obj(obj_name,
              vertices,
              faces,
              texture_name = "texture.jpg",
              colors=None,
              texture=None,
              uvcoords=None,
              uvfaces=None,
              inverse_face_order=False,
              normal_map=True,
              ):

    if os.path.splitext(obj_name)[-1] != '.obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name
    material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        # f.write('# %s\n' % os.path.basename(obj_name))
        # f.write('#\n')
        # f.write('\n')
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
            f.write('usemtl %s\n' % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write('f {}/{} {}/{} {}/{}\n'.format(
                    #  faces[i, 2], uvfaces[i, 2],
                    #  faces[i, 1], uvfaces[i, 1],
                    #  faces[i, 0], uvfaces[i, 0]
                    faces[i, 0], uvfaces[i, 0],
                    faces[i, 1], uvfaces[i, 1],
                    faces[i, 2], uvfaces[i, 2]
                )
                )
            # write mtl
            with open(mtl_name, 'w') as f:
                f.write('newmtl %s\n' % material_name)
                s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
                f.write(s)

                if normal_map is not None:
                    name, _ = os.path.splitext(obj_name)
                    normal_name = f'{name}_normals.png'
                    f.write(f'disp {normal_name}')
                    # out_normal_map = normal_map / (np.linalg.norm(
                    #     normal_map, axis=-1, keepdims=True) + 1e-9)
                    # out_normal_map = (out_normal_map + 1) * 0.5

                    cv2.imwrite(
                        normal_name,
                        # (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
                        normal_map
                    )
            #save texture, obj, uv 
            
            skimage.io.imsave(texture_name, texture)

def normalize_keypoints(keypoints3d):
    center = keypoints3d[0]
    keypoints3d = keypoints3d - center
    axis1 = keypoints3d[165] - keypoints3d[391]
    axis2 = keypoints3d[2] - keypoints3d[0]
    axis3 = np.cross(axis2,axis1)
    axis3 = axis3/np.linalg.norm(axis3)
    axis2 = axis2/np.linalg.norm(axis2)
    axis1 = np.cross(axis3, axis2)
    axis1 = axis1/np.linalg.norm(axis1)
    U = np.array([axis3,axis2,axis1])
    keypoints3d = keypoints3d.dot(U)
    keypoints3d = keypoints3d - keypoints3d.mean(axis=0)
    return keypoints3d

def ColorDistance(rgb1,rgb2):
    #removed alpha channel
    rgb1 = rgb1[:3]
    rgb2 = rgb2[:3]
    '''d = {} distance between two colors(3)'''
    rm = 0.5*(rgb1[0]+rgb2[0])
    d = sum((2+rm,4,3-rm)*(rgb1-rgb2)**2)**0.5
    return d

def initialize_dlib(facial_landmark_predictor:str):
    print('Loading facial landmark predictor...')
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor()
    return detector, predictor

def extract_faces_landmarks(img, detector, predictor):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray scale frame
    faces = detector(img_gray, 0)

    for idx, face in enumerate(faces):
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        yield {
                "face": face
                , "landmarks": landmarks_points
        }

def mask_landmark(img,pts):
    # Create a mask
    mask = np.ones(img.shape[:2],np.uint8) #np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask,[pts],-1,(0,0,0),-1,cv2.LINE_AA)
    masked_img = cv2.bitwise_and(img,img,mask=mask)
    return masked_img

if __name__ == "__main__":
    uv_path = "./data/uv_map.json" #taken from https://github.com/spite/FaceMeshFaceGeometry/blob/353ee557bec1c8b55a5e46daf785b57df819812c/js/geometry.js
    uv_map_dict = json.load(open(uv_path))
    uv_map = np.array([ (uv_map_dict["u"][str(i)],uv_map_dict["v"][str(i)]) for i in range(468)])
    img_path = r"C:\Users\joeli\OneDrive\Documents\GitHub\mediapipe-facemesh\images\neutral.jpg"
    img_ori = skimage.io.imread(img_path)
    # imshow(img_ori)
    img = img_ori
    # img = masked_image
    H,W,_ = img.shape
    #run facial landmark detection
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.6) as face_mesh:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img)

    assert len(results.multi_face_landmarks)==1 

    face_landmarks = results.multi_face_landmarks[0]
    keypoints = np.array([(W*point.x,H*point.y) for point in face_landmarks.landmark[0:468]])#after 468 is iris or something else
    ax = imshow(PIL.Image.fromarray(img_ori).convert('RGB'))
    ax.plot(keypoints[:, 0], keypoints[:, 1], '.b', markersize=2)
    # plt.show()
    H_new,W_new = 512,512
    keypoints_uv = np.array([(W_new*x, H_new*y) for x,y in uv_map])

    tform = PiecewiseAffineTransform()
    tform.estimate(keypoints_uv,keypoints)
    texture = warp(img_ori, tform, output_shape=(H_new,W_new))
    texture = (255*texture).astype(np.uint8)
    # Assuming texture is your image array...
    if texture.shape[2] == 4:   # If the image has an alpha channel...
        texture = texture[:, :, :3]  # ...remove the alpha channel
    # ax = imshow(texture)
    ax.plot(keypoints_uv[:, 0], keypoints_uv[:, 1], '.b', markersize=2)
    # plt.show()

    keypoints3d = np.array([(point.x,point.y,point.z) for point in face_landmarks.landmark[0:468]])
    obj_filename = r"C:\Users\joeli\OneDrive\Documents\GitHub\mediapipe-facemesh\data\canonical_face_model.obj"
    # obj_filename = "/Users/joeljohnson/Desktop/mediapipe-facemesh/data/head_template.obj"
    verts,uvcoords,faces,uv_faces = load_obj(obj_filename)


    # Assuming texture is your image array...
    if texture.shape[2] == 4:   # If the image has an alpha channel...
        texture = texture[:, :, :3]  # ...remove the alpha channel
    vertices = normalize_keypoints(keypoints3d)

    # borrowed from https://github.com/YadiraF/PRNet/blob/master/utils/write.py
    path = r"C:\Users\joeli\OneDrive\Documents\GitHub\mediapipe-facemesh\images\neutral"
    obj_name =  os.path.join(path, "neutral.obj")
    texture_name = os.path.join(path, "neutral.png")
    
    # obj_name2 = "/Users/joeljohnson/Desktop/mediapipe-facemesh/data/head_template.obj"
    write_obj(obj_name,
                vertices,
                faces,
                texture_name = texture_name,
                texture=texture,
                uvcoords=uvcoords,
                uvfaces=uv_faces,
                )



    colors = ['red','green','blue','yellow','purple','orange','pink','brown','black','white']
    pixels = {}
    for i in range(len(keypoints)-2):
        c = colors[i%len(colors)]
        pixels[i] = img_ori[int(keypoints[i,1]),int(keypoints[i,0])]
        ax.text(keypoints[i,0], keypoints[i,1], str(i), color=c)
    
    # Get the current working directory
    current_dir = os.getcwd()
    # Create a temporary file name with full path

    try:
        temp_file1 = tempfile.NamedTemporaryFile(suffix=".png", dir=current_dir, delete=False)

        texture_image = Image.fromarray(texture)
        # Save the texture image to the temporary file
        texture_image.save(temp_file1.name)
        
        # Load the 3D model using vtkplotter
        mesh = load(obj_name)
        #smooth the mesh
        
        # Load and apply the texture image from the temporary file
        mesh.texture(temp_file1.name)
        #make the view straight on
        mesh.rotateY(0)
        mesh.rotateX(0)
        # Show the mesh
        mesh.show()

    except IOError:
        print("Unexpected error:", str(IOError))
    except PermissionError:
        print("Permission denied: " + str(PermissionError))
    except:
        print("Unexpected error")
    finally:
        # Delete the temporary file
        os.remove(temp_file1.name)


        

# %%
