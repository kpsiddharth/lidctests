import dicom
import pylab
from numpy import shape
import os
import numpy

path = 'D:\\Data\\ml\\medical\\LIDC\\DICOM\\DOI\\'
large_count = 0
small_count = 0

STANDARD_HEIGHT = 512
STANDARD_LENGTH = 512

for root, dirs, files in os.walk(path):
    for name in files:
        if name.endswith('dcm'):
            fullpath = root + '\\' + name
            ds = dicom.read_file(fullpath)
            if numpy.shape(ds.pixel_array)[0] > 2000:
                large_count += 1
            else:
                small_count += 1
                pixel_array = numpy.shape(ds._pixel_data_numpy())
                if pixel_array[0] == STANDARD_HEIGHT and pixel_array[1] == STANDARD_LENGTH:
                    pass
                    pylab.imshow(ds.pixel_array, cmap = pylab.cm.bone)
                    pylab.show()
                else:
                    print ('Shape = ', numpy.shape(ds._pixel_data_numpy()))
                pass

print ('X-Ray Count = ', large_count)
print ('CT Scan Count = ', small_count)
#print type(ds.dir('ReferringPhysicianName')[0])
