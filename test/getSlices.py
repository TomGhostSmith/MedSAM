import os
import json
import numpy
from image import Image
import imageio

from tqdm import tqdm

# def mergeXSlice(image, x, slice):
#     image[x, :, :] = slice
#     return image

# def mergeYSlice(image, y, slice):
#     image[:, y, :] = slice
#     return image

# def mergeZSlice(image, z, slice):
#     image[:, :, z] = slice
#     return image

def generateSlices(originImage:Image, maskImage:Image, resultImage:Image, filename:str, isTest:bool, dimension):
    margin = 1
    setting = {
        'x': (0, originImage.getXSlice, maskImage.getXSlice, resultImage.getXSlice),
        'y': (1, originImage.getYSlice, maskImage.getYSlice, resultImage.getYSlice),
        'z': (2, originImage.getZSlice, maskImage.getZSlice, resultImage.getZSlice)
    }

    axisIndex, getSlice, getMaskSlice, getResultSlice = setting[dimension]

    if (isTest):
        targetFolder = '/Data/MedicalImageProcessing/MSD/Task06_Lung/Slices/test'
    else:
        targetFolder = '/Data/MedicalImageProcessing/MSD/Task06_Lung/Slices/train'
    
    # originSliceFolder = f"{targetFolder}/image"

    for i in range(originImage.shape[axisIndex]):
        slice = numpy.array(getSlice(i))

        if (numpy.max(slice) > numpy.min(slice)):
            slice = (slice - numpy.min(slice)) / (numpy.max(slice) - numpy.min(slice)) * 255
        else:
            slice = numpy.zeros_like(slice)
        slice = slice.astype(numpy.uint8)
        mask = numpy.array(getMaskSlice(i)).copy()
        res = numpy.array(getResultSlice(i)).copy().astype(numpy.uint8)

        rows, cols = numpy.where(mask == 1)
        if (len(cols) > 0 and len(rows) > 0):
            xMin = max(numpy.min(cols) - margin, 0)
            xMax = min(numpy.max(cols) + margin, mask.shape[0])
            yMin = max(numpy.min(rows) - margin, 0)
            yMax = min(numpy.max(rows) + margin, mask.shape[1])
            box = [int(xMin), int(yMin), int(xMax), int(yMax)]

            imageio.imwrite(f"{targetFolder}/image/{filename}_{dimension}_{i}.png", slice)
            with open(f"{targetFolder}/box/{filename}_{dimension}_{i}.json", 'wt') as fp:
                json.dump(box, fp)
            imageio.imwrite(f"{targetFolder}/label/{filename}_{dimension}_{i}.png", res)

            # print(f'{targetFolder}/box/{filename}_{dimension}_{i}.json')

def processAll():
    testset = {'lung_001.nii.gz', 'lung_003.nii.gz', 'lung_025.nii.gz', 'lung_027.nii.gz', 'lung_044.nii.gz', 'lung_046.nii.gz', 'lung_048.nii.gz', 'lung_058.nii.gz', 'lung_062.nii.gz', 'lung_070.nii.gz', 'lung_081.nii.gz', 'lung_083.nii.gz', 'lung_093.nii.gz'}
    originFolder = '/Data/MedicalImageProcessing/MSD/Task06_Lung/imagesTr'            # e.g. lung_001.nii.gz
    resultFolder = '/Data/MedicalImageProcessing/MSD/Task06_Lung/labelsTr'            # e.g. lung_001.nii.gz
    maskFolder = '/Data/MedicalImageProcessing/MSD/Task06_Lung/results/UnetrPP_All'   # e.g. lung_001.nii.gz

    for file in tqdm(sorted(os.listdir(originFolder))):
        if (file.endswith('.nii.gz')):
            origin = Image.loadFromFile(f"{originFolder}/{file}")
            result = Image.loadFromFile(f"{resultFolder}/{file}")
            mask = Image.loadFromFile(f"{maskFolder}/{file}")
            generateSlices(origin, result, mask, file[:-7], file in testset, 'x')
            generateSlices(origin, result, mask, file[:-7], file in testset, 'y')
            generateSlices(origin, result, mask, file[:-7], file in testset, 'z')

def main():
    processAll()


main()