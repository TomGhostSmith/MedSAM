import os
import sys
import json
import nibabel
import numpy
import pandas

sys.path.append('.')
from image import Image
import ParaUtils

def getStatistics(groundTruth:Image, prediction:Image):
    groundTruth = numpy.array(groundTruth.data)
    prediction = numpy.array(prediction.data)
    intersection = numpy.sum((groundTruth == 1) & (prediction == 1))
    union = numpy.sum((groundTruth == 1) | (prediction == 1))
    total = numpy.sum(groundTruth == 1) + numpy.sum(prediction == 1)

    IoU = intersection / union if union != 0 else 0

    dice = (2 * intersection) / total if total != 0 else 0

    return IoU, dice

def evaluateModel(modelName, args=None):
    basePath = "/Data/MedicalImageProcessing/MSD/Task06_Lung/results"
    gtPath = f"{basePath}/std"
    if (args is not None):
        modelPath = f"{basePath}/{modelName}/results_{args['name']}"
    else:
        modelPath = f"{basePath}/{modelName}"

    files = os.listdir(gtPath)
    results = numpy.zeros((len(files), 2))
    for idx, file in enumerate(files):
        groundTruth = Image.loadFromFile(f"{gtPath}/{file}")
        prediction = Image.loadFromFile(f"{modelPath}/{file}")
        res = getStatistics(groundTruth, prediction)
        results[idx] = res
    
    result = numpy.average(results, axis=0)
    return result

def evaluateModels(models):
    fullArgs = list()
    for margin in [0, 1, 3, 5]:
        for pool in ["min", "ave", "max"]:
            for dilation in [True, False]:
                fullArgs.append({
                    "dilation": dilation,
                    "pool": pool,
                    "margin": margin,
                    "name": f"pool={pool}_margin={margin}_dilation={dilation}"
                })
    
    resultLines = list()
    resultLines.append("model,margin,pool,postprocess,IoU,Dice\n")

    modelList = list()
    marginList = list()
    poolList = list()
    postprocessList = list()
    paramList = list()
    for model in models:
        if model.endswith('MedSAM'):
            for args in fullArgs:
                paramList.append((model, args))
                modelList.append(model)
                marginList.append(args['margin'])
                poolList.append(args['pool'])
                postprocessList.append(args['dilation'])
        else:
            paramList.append((model, None))
            modelList.append(model)
            marginList.append('N/A')
            poolList.append('N/A')
            postprocessList.append('N/A')

    results = ParaUtils.parallelWithResult(paramList, evaluateModel, progressBar=True)
    # results = [evaluateModel(*p) for p in paramList]
    IoUList, diceList = zip(*results)

    df = pandas.DataFrame({
        "model": modelList,
        "margin": marginList,
        "pool": poolList,
        "postprocess": postprocessList,
        "IoU": IoUList,
        "Dice": diceList
    })

    df.to_csv('./result.csv', index=False)
    
    
def main():
    models = [
        "UnetrPP_raw",
        "UnetrPP_postprocessed",
        "nnUNet",
        "UnetrPP_raw+MedSAM",
        "UnetrPP_postprocessed+MedSAM",
        "nnUNet+MedSAM",
        "std+MedSAM"
    ]
    evaluateModels(models)

main()