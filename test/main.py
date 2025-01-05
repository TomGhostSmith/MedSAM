import os
import sys
import torch
import numpy
import nibabel
import imageio
import multiprocessing

from scipy.ndimage import binary_dilation, binary_erosion

sys.path.append('./test')
from image import Image
import ParaUtils


def mergeXSlice(image, x, slice):
    image[x, :, :] = slice
    return image

def mergeYSlice(image, y, slice):
    image[:, y, :] = slice
    return image

def mergeZSlice(image, z, slice):
    image[:, :, z] = slice
    return image



def processOneDimention(fileName, modelShortName,originImage:Image, maskImage:Image, dimension, margin, device='cuda:0'):
    print(f"Handling {dimension}-axis on {device}")
    setting = {
        'x': (0, originImage.getXSlice, maskImage.getXSlice, mergeXSlice),
        'y': (1, originImage.getYSlice, maskImage.getYSlice, mergeYSlice),
        'z': (2, originImage.getZSlice, maskImage.getZSlice, mergeZSlice)
    }

    axisIndex, getSlice, getMaskSlice, mergeSlice = setting[dimension]

    resultImage = numpy.zeros(originImage.shape)
    # sliceFolder = f"./test/slices_{os.getpid()}"
    # maskFolder = f"./test/masks_{os.getpid()}"

    sliceFolder = f"/Data/MedicalImageProcessing/MedSAM/slices/{modelShortName}/{fileName}"
    maskFolder = f"/Data/MedicalImageProcessing/MedSAM/masks/{modelShortName}/{fileName}"

    os.makedirs(sliceFolder, exist_ok=True)
    os.makedirs(maskFolder, exist_ok=True)

    for i in range(originImage.shape[axisIndex]):
        slice = numpy.array(getSlice(i))

        if (numpy.max(slice) > numpy.min(slice)):
            slice = (slice - numpy.min(slice)) / (numpy.max(slice) - numpy.min(slice)) * 255
        else:
            slice = numpy.zeros_like(slice)
        slice = slice.astype(numpy.uint8)
        mask = numpy.array(getMaskSlice(i)).copy()

        rows, cols = numpy.where(mask == 1)
        if (len(cols) > 0 and len(rows) > 0):
            xMin = max(numpy.min(cols) - margin, 0)
            xMax = min(numpy.max(cols) + margin, mask.shape[0])
            yMin = max(numpy.min(rows) - margin, 0)
            yMax = min(numpy.max(rows) + margin, mask.shape[1])

            if (not os.path.exists(f"{maskFolder}/seg_{dimension}_{i}.png")):
                imageio.imwrite(f"{sliceFolder}/{dimension}_{i}.png", slice)
                os.system(f'python MedSAM_Inference.py -i {sliceFolder}/{dimension}_{i}.png -o {maskFolder} --box "{[xMin, yMin, xMax, yMax]}" --device {device}')

            # load masked image and copy to the result Image
            sliceResult = imageio.v3.imread(f"{maskFolder}/seg_{dimension}_{i}.png")
            mergeSlice(resultImage, i, sliceResult)  # * 255?
        else:
            pass  # do nothing, because the resultImage is initialized with zeros

    # remove existing files
    # for file in os.listdir(sliceFolder):
    #     os.remove(f"{sliceFolder}/{file}")
    # for file in os.listdir(maskFolder):
    #     os.remove(f"{maskFolder}/{file}")
    # os.removedirs(sliceFolder)
    # os.removedirs(maskFolder)
    
    return resultImage

def mergeResult(results, idx, result):
    results[idx] = result

def processOne(fileName, originFile, unetrResultFile, savePath, args):
    print(f"segment {originFile}")
    originImage = Image.loadFromFile(originFile)
    maskImage = Image.loadFromFile(unetrResultFile)
    print(originImage.data.shape)

    # results = numpy.zeros((3, *originImage.data.shape))

    params = [
        (fileName, args['shortName'], originImage, maskImage, 'x', args['margin'], 'cuda:0'),
        (fileName, args['shortName'], originImage, maskImage, 'y', args['margin'], 'cuda:1'),
        (fileName, args['shortName'], originImage, maskImage, 'z', args['margin'], 'cuda:1')
    ]


    results = ParaUtils.parallelWithResult(params, processOneDimention, maxThreads=3)
    # x = processOneDimention(originImage, maskImage, 'x', args['margin'], 'cuda:0')

    # with multiprocessing.Pool(processes=3) as pool:
    #     results = pool.starmap(processOneDimention, params)

    if (args['dilation']):  # perform dilation and erosion for better continuity
        postProcessedResult = numpy.zeros_like(results)
        kernel = numpy.array([
            [[0, 1, 0], 
             [1, 1, 1], 
             [0, 1, 0]],

            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],

            [[0, 1, 0], 
             [1, 1, 1], 
             [0, 1, 0]]
        ])
        for idx, res in enumerate(results):
            dilatedRes = binary_dilation(res, kernel)
            erodedRes = binary_erosion(dilatedRes, kernel)
            postProcessedResult[idx] = erodedRes.astype(numpy.float32)
        results = postProcessedResult
            

    if (args['pool'] == 'min'):
        resultImage = numpy.min(results, axis=0)
    elif (args['pool'] == 'ave'):
        resultImage = numpy.round(numpy.average(results, axis=0))
    elif (args['pool'] == 'max'):
        resultImage = numpy.max(results, axis=0)
        

    nibabel.save(Image.loadFromData(resultImage, originImage.affine).nii, savePath)

def processAll(args):
    # imageFolder = "/Data/MedicalImageProcessing/UnetrPP/DATASET_Lungs/unetr_pp_raw/unetr_pp_raw_data/Task06_Lung/imagesTr"
    imageFolder = args['imgPath']
    saveFolder = f"{args['savePath']}/results_{args['name']}"
    UnetrResultFolder = args['boxPath']
    if (os.path.exists(saveFolder)):
        ParaUtils.showInfo(f"Skip config: {args['name']}")
        return
    print(f"processing config: {args['name']}")
    os.makedirs(saveFolder)
    files = sorted(os.listdir(UnetrResultFolder))

    fileList = list()
    paramList = list()
    for file in files:
        if (file.endswith('.nii.gz')):
            fileList.append(file)
            paramList.append((file, f"{imageFolder}/{file}", f"{UnetrResultFolder}/{file}", f"{saveFolder}/{file}", args))
    # with multiprocessing.Pool(processes=1) as pool:
    #     for file in fileList:
    #         pool.apply_async(processOne, (f"{imageFolder}/{file}", f"{UnetrResultFolder}/{file}", f"{saveFolder}/{file}", args))
    #     pool.close()
    #     pool.join()

    # with multiprocessing.Pool(processes=4) as pool:
    #     pool.starmap(processOne, paramList)

    ParaUtils.parallel(paramList, processOne, maxThreads=4)

    # for file in fileList:
    #     processOne(f"{imageFolder}/{file}", f"{UnetrResultFolder}/{file}", f"{saveFolder}/{file}", args)

def main():
    ParaUtils.showInfo("main")

    # model name, origin img folder, box result folder and save folder
    pathLists = [
        ("UnetrPP-raw", "/Data/MedicalImageProcessing/MSD/Task06_Lung/imagesTr", "/Data/MedicalImageProcessing/MSD/Task06_Lung/results/UnetrPP_raw", "/Data/MedicalImageProcessing/MSD/Task06_Lung/results/UnetrPP_raw+MedSAM"),
        ("UnetrPP-postprocessed", "/Data/MedicalImageProcessing/MSD/Task06_Lung/imagesTr", "/Data/MedicalImageProcessing/MSD/Task06_Lung/results/UnetrPP_postprocessed", "/Data/MedicalImageProcessing/MSD/Task06_Lung/results/UnetrPP_postprocessed+MedSAM"),
        ("nnUNet", "/Data/MedicalImageProcessing/MSD/Task06_Lung/imagesTr", "/Data/MedicalImageProcessing/MSD/Task06_Lung/results/nnUNet", "/Data/MedicalImageProcessing/MSD/Task06_Lung/results/nnUNet+MedSAM"),
        ("std", "/Data/MedicalImageProcessing/MSD/Task06_Lung/imagesTr", "/Data/MedicalImageProcessing/MSD/Task06_Lung/results/std", "/Data/MedicalImageProcessing/MSD/Task06_Lung/results/std+MedSAM")
    ]
    argsList = list()
    for modelName, imgPath, boxPath, savePath in pathLists:
        for pool in ["min", "ave", "max"]:
            # for margin in [1, 3, 5]:
            for margin in [0]:
                for dilation in [True, False]:
                    argsList.append({
                        "dilation": dilation,
                        "imgPath": imgPath,
                        "boxPath": boxPath,
                        "savePath": savePath,
                        "pool": pool,
                        "margin": margin,
                        "shortName": f"model={modelName}_margin={margin}",
                        "name": f"pool={pool}_margin={margin}_dilation={dilation}"
                    })
    for args in argsList:
        processAll(args)


if (__name__ == '__main__'):
    main()