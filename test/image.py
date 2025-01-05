import numpy
import torch
import nibabel
from TPTBox import NII
import matplotlib.pyplot as plt

class Image():
    def __init__(self) -> None:
        self.affine = None
        self.invAffine = None
        self.shape = None
        self.data = None
        self.nii = None

    @classmethod
    def loadFromData(cls, data, affine=numpy.eye(4)):
        i = cls()
        i.data = torch.tensor(data)
        i.affine = affine
        i.invAffine = numpy.linalg.inv(affine)
        i.shape = data.shape
        i.nii = nibabel.Nifti1Image(data, affine)
        return i

    @classmethod
    def loadFromFile(cls, filePath):
        i = cls()
        image = nibabel.load(filePath)
        data = image.get_fdata()
        affine = image.affine
        nii = NII.load(filePath, False)

        i.data = torch.tensor(data)
        i.affine = affine
        i.shape = data.shape
        i.invAffine = numpy.linalg.inv(affine)
        i.nii = nii
        return i

    def getImageCoordinate(self, realWorldCoordinate):
        realWorldCoordinate = numpy.append(realWorldCoordinate, 1)
        imageCoordinate = self.invAffine @ realWorldCoordinate
        imageCoordinate = imageCoordinate[0:3]
        # imageCoordinate = numpy.array([imageCoordinate[1], imageCoordinate[0], imageCoordinate[2]])
        return numpy.rint(imageCoordinate).astype(int)
    
    def getRealWorldCoordinate(self, imageCoordinate):
        # imageCoordinate = numpy.array([imageCoordinate[1], imageCoordinate[0], imageCoordinate[2], 1])
        imageCoordinate = numpy.append(imageCoordinate, 1)
        realWorldCoordinate = self.affine @ imageCoordinate
        return realWorldCoordinate[0:3]
    
    def getXSlice(self, index=None, proportion=None):
        if (index != None):
            return self.data[index, :, :]
        elif (proportion != None):
            return self.data[int(self.shape[0] * proportion), :, :]
        else:
            return None
        
    def getYSlice(self, index=None, proportion=None):
        if (index != None):
            return self.data[:, index, :]
        elif (proportion != None):
            return self.data[:, int(self.shape[1] * proportion), :]
        else:
            return None
        
    def getZSlice(self, index=None, proportion=None):
        if (index != None):
            return self.data[:, :, index]
        elif (proportion != None):
            return self.data[:, :, int(self.shape[2] * proportion)]
        else:
            return None